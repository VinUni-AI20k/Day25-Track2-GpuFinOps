# Lab 25 Write-up — NimbusAI GPU Cost Optimization

## 1. Baseline vs. Optimized

| | Monthly spend | $/1M-token (inference only) |
|---|---|---|
| Baseline | $27,133 | $6.488 |
| Optimized | $14,626 | $1.126 |
| **Savings** | **$12,507 (46%)** | **82.6%** on the inference slice |

`verify.py` 11/11 and `pytest -q` 15/15 both pass on the unmodified engine plus the
extensions below — none of the automated checks were relaxed to get there.

## 2. Per-lever analysis

| Lever | Monthly $ saved | Why |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | Biggest absolute dollar lever — 3 of 8 workloads are steady 24/7 inference jobs eligible for a 45%-off reserved commitment, and the interruptible training jobs move to spot at ~40% off with checkpoint rework priced in. |
| Inference (cascade/cache/batch) | $1,212/day-equivalent → largest **relative** lever at 82.6% off $/1M-token | Cascading cheap requests to the small model, caching the ~300x-reused system prompts at -90%, and batching non-real-time calls compound multiplicatively (0.5 × 0.1 ≈ 0.05 of naive cost for the fully-stacked case). |
| Right-size util-lies | $655 | Downgrading the two GPUs that report high `gpu_util_pct` but low MFU one tier down (H100→A100, A10G→L4) recovers most of the wasted headroom without touching throughput. |
| Kill idle GPUs | $600 | `gpu-h100-5` sits idle 8h — a zero-risk, same-day fix. |

**Which lever matters most?** Purchasing wins on absolute dollars because NimbusAI's
bill is dominated by 24/7 production inference fleets, where a duty cycle
committed once compounds every month. Inference levers win on **unit economics**
($/1M-token) — a 82.6% cut there means every future volume increase is 5.75x
cheaper, which purchasing decisions alone can't deliver.

## 3. The GPU-Util Lie

`gpu-h100-4` reports **98.2% GPU-Util** but only **19.4% MFU** — it's "busy" almost
all the time, but doing barely a fifth of the FLOPs an H100 is capable of.
`nvidia-smi`'s util metric only measures whether the SM clock has *any* work
queued each sampling tick; it can't tell whether that work is a full tensor-core
matmul or the GPU stalling on an HBM read waiting for the next kernel launch. A
memory-bound decode loop, small batch sizes, or Python-side dispatch overhead all
produce this pattern: 98% "busy", ~20% actual throughput. Financially this means
NimbusAI has been paying the **full H100-hour rate for a fifth of the compute** —
right-sizing this one GPU down to an A100 alone recovers a meaningful slice of
the $655/mo right-sizing lever, at equivalent effective throughput.

## 4. "Your Turn" extensions implemented

### Extension 1 — `recommend_tier()` with GPU interrupt rate + 1yr/3yr duration
Added a `GPU_INTERRUPT_RATE` table (H100/H200 ~3%/h reclaim, A100 5%, A10G/L4 8%)
and a `job_days` signal so persistent (≥25 days/month) reserved-eligible jobs get
the deeper 3yr commitment while shorter-lived ones get 1yr, and interruptible jobs
on high-reclaim GPUs at near-continuous duty cycle get pushed to `reserved_1yr`
instead of spot (checkpoint rework would eat the discount).
**Measured:** on NimbusAI's actual 8-job fleet, `savings_pct` is unchanged at
39.1% — no current job happens to combine a high-reclaim GPU with high duty
cycle, or a <25-day reserved-eligible run. Running the new policy across a
synthetic matrix (GPU type × hours/day × interruptible, `missions/m3_purchasing.py`
"Extension 1" section) shows it *does* diverge: A10G/L4 at 20h/day+interruptible
flips from `spot`→`reserved_1yr`. **Insight:** the base policy silently assumes
uniform reclaim risk across GPU types — this fleet just doesn't happen to expose
that blind spot yet, but it would as soon as NimbusAI adds A10G/L4 training capacity.

### Extension 3 — `cache_is_worth_it()`
Added a break-even check: caching only pays once `avg_cache_reads * (1 - read_discount)
> write_cost_per_m`. Applied per-request in M2 using each row's real
(team, project) prefix-reuse count from `token_usage.csv` and an assumed flat
$0.15/M cache-storage fee.
**Measured:** the real average reuse count is **~300 reads per shared system
prompt** (only 4 team/project prefixes across 2,400 requests), versus a
break-even of 0.833 reads (small tier) / 0.056 reads (large tier) — caching
clears break-even by 2–3 orders of magnitude, so 0/2,400 requests get gated off.
**Insight:** at NimbusAI's traffic concentration, caching is essentially free
money; the check would only start rejecting requests for genuinely one-off,
un-reused prompts, which is exactly the risk `cache_is_worth_it()` is meant to
catch before it ships to a lower-traffic team.

### Extension 4 — Reasoning budget
Split `$` and `Wh` between `is_reasoning=1` and `=0` traffic in M2 and surfaced it
in the M5 report.
**Measured:** reasoning is only **8.4% of requests** but **16.5% of inference
spend** and **94% of energy** (consistent with the ~80x-per-query energy
multiplier). A proposed routing rule — only invoke reasoning below a
confidence-score threshold, capping it to 5% of traffic — saves **$0.56/day
(~$17/mo)** and **~12,000 Wh/day (~360 kWh/mo)**. **Insight:** reasoning's
energy footprint is wildly disproportionate to its request share, so it's the
single highest-leverage sustainability lever even though it's a modest dollar one.

## 5. Recommendations for NimbusAI (first 3 actions)

1. **Kill the idle GPU today.** `gpu-h100-5`'s 8h/day idle window is a zero-risk,
   zero-migration $600/mo recovery — ship it before anything else.
2. **Ship cascade + cache + batch on the inference path.** This is the highest-ROI
   engineering lever (82.6% off $/1M-token) and, unlike purchasing changes, it
   compounds with every future volume increase rather than being a one-time
   contract negotiation.
3. **Re-negotiate purchasing tiers using duty cycle, not vibes**: move the three
   24/7 inference fleets to reserved and the interruptible training/dev jobs to
   spot+checkpoint — this is the single biggest absolute-dollar lever ($10,040/mo)
   and is now backed by a GPU-type-aware break-even policy (Extension 1) instead
   of a flat rule of thumb.
4. *(Sustainability follow-up)* Cap reasoning traffic behind a confidence
   threshold and prefer `europe-north1` for any relocatable interruptible
   training job — it's both the cheapest and cleanest region in the catalog
   (30 gCO2/kWh vs. 660 in `europe-central2`).
