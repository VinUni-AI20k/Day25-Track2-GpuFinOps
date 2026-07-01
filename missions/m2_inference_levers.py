"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Extension 3 assumption: a flat cache-storage fee ($/M-token, Gemini-style TTL
# storage rather than Anthropic's write-premium) — the same absolute fee bites a
# cheap tier's margin harder than an expensive tier's.
CACHE_WRITE_FEE_PER_M = 0.15


def _avg_cache_reads_by_project(rows: list[dict]) -> dict:
    """Proxy for 'how many times is this cached prefix re-read': count of cached
    requests sharing the same (team, project), i.e. the same shared system prompt."""
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            key = (r["team"], r["project"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    reuse_counts = _avg_cache_reads_by_project(rows)

    base_cost = opt_cost = 0.0
    total_tokens = 0
    reasoning_cost = reasoning_tokens = 0.0
    normal_cost = normal_tokens = 0.0
    cache_gated_off = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r.get("is_reasoning", 0))))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (gated by cache_is_worth_it),
        # batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        avg_reads = reuse_counts.get((r["team"], r["project"]), 0)
        write_frac = CACHE_WRITE_FEE_PER_M / pin  # fee expressed as a fraction of this tier's price
        use_cache = cached > 0 and pricing.cache_is_worth_it(avg_reads, write_frac)
        if not use_cache:
            cache_gated_off += 1
        row_cost = pricing.request_cost(
            inp, out, pin, pout, cached_in=cached if use_cache else 0, batch=is_batch)
        opt_cost += row_cost
        # Extension 4: reasoning traffic cost/energy budget
        if is_reasoning:
            reasoning_cost += row_cost
            reasoning_tokens += inp + out
        else:
            normal_cost += row_cost
            normal_tokens += inp + out

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

    ext3 = _extension3_cache_worth_it(reuse_counts, cache_gated_off, len(rows), verbose=verbose)
    ext4 = _extension4_reasoning_budget(rows, reasoning_cost, reasoning_tokens, normal_cost,
                                         normal_tokens, opt_cost, verbose=verbose)

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "extension3": ext3, "extension4": ext4,
    }


def _extension3_cache_worth_it(reuse_counts: dict, gated_off: int, n_rows: int, verbose: bool = True) -> dict:
    """Your Turn Extension 3 — cache_is_worth_it() break-even per model tier,
    checked against the real avg re-read count observed in token_usage.csv."""
    avg_reads = sum(reuse_counts.values()) / len(reuse_counts) if reuse_counts else 0.0
    per_tier = {}
    for tier, (pin, _pout) in MODEL_PRICES.items():
        write_frac = CACHE_WRITE_FEE_PER_M / pin
        be_reads = pricing.break_even_cache_reads(write_frac)
        worth_it = pricing.cache_is_worth_it(avg_reads, write_frac)
        per_tier[tier] = {"break_even_reads": round(be_reads, 3), "worth_it": worth_it}
    if verbose:
        print("\n-- Extension 3: cache_is_worth_it() (flat $%.2f/M cache-storage fee) --" % CACHE_WRITE_FEE_PER_M)
        print(f"avg re-reads per shared (team,project) prefix in data: {avg_reads:.1f}")
        for tier, d in per_tier.items():
            print(f"  {tier:6} tier: break-even at {d['break_even_reads']:.3f} reads -> "
                  f"worth it? {d['worth_it']}")
        print(f"requests where cache was gated OFF (not worth it or no cache hit): {gated_off}/{n_rows}")
    return {"avg_cache_reads": round(avg_reads, 1), "per_tier": per_tier, "gated_off": gated_off}


def _extension4_reasoning_budget(rows, reasoning_cost, reasoning_tokens, normal_cost,
                                  normal_tokens, opt_cost, verbose: bool = True) -> dict:
    """Your Turn Extension 4 — split $ and Wh spend between is_reasoning and normal
    traffic, and estimate savings from capping reasoning to 10% of traffic."""
    from finops.sustainability import wh_per_query

    n_reasoning = sum(1 for r in rows if int(num(r.get("is_reasoning", 0))) == 1)
    n_total = len(rows)
    reasoning_share_traffic = n_reasoning / n_total * 100 if n_total else 0.0
    reasoning_share_cost = reasoning_cost / opt_cost * 100 if opt_cost else 0.0

    wh_reasoning = sum(
        wh_per_query(int(num(r["input_tokens"])) + int(num(r["output_tokens"])), is_reasoning=True)
        for r in rows if int(num(r.get("is_reasoning", 0))) == 1)
    wh_normal = sum(
        wh_per_query(int(num(r["input_tokens"])) + int(num(r["output_tokens"])), is_reasoning=False)
        for r in rows if int(num(r.get("is_reasoning", 0))) == 0)

    # Routing rule: only invoke reasoning below a low-confidence-score threshold,
    # capping it to 5% of requests (route the rest to the cascaded normal tiers);
    # estimate $/Wh saved proportionally.
    cap_frac = 0.05
    if n_reasoning > 0:
        keep_frac = min(1.0, (cap_frac * n_total) / n_reasoning)
    else:
        keep_frac = 1.0
    cost_saved = reasoning_cost * (1 - keep_frac)
    wh_saved = wh_reasoning * (1 - keep_frac)

    if verbose:
        print("\n-- Extension 4: Reasoning budget (is_reasoning traffic) --")
        print(f"reasoning traffic: {n_reasoning}/{n_total} requests ({reasoning_share_traffic:.1f}%)")
        print(f"reasoning cost share: ${reasoning_cost:,.2f}/day of ${opt_cost:,.2f}/day ({reasoning_share_cost:.1f}%)")
        print(f"energy: reasoning {wh_reasoning:,.1f} Wh/day vs normal {wh_normal:,.1f} Wh/day "
              f"({(wh_reasoning / (wh_reasoning + wh_normal) * 100) if (wh_reasoning + wh_normal) else 0:.1f}% of energy "
              f"from {reasoning_share_traffic:.1f}% of traffic)")
        print(f"routing rule: cap reasoning at {cap_frac:.0%} of traffic -> "
              f"save ${cost_saved:,.2f}/day and {wh_saved:,.1f} Wh/day")

    return {
        "n_reasoning": n_reasoning, "n_total": n_total,
        "reasoning_share_traffic_pct": round(reasoning_share_traffic, 1),
        "reasoning_share_cost_pct": round(reasoning_share_cost, 1),
        "wh_reasoning": round(wh_reasoning, 1), "wh_normal": round(wh_normal, 1),
        "capped_cost_saved": round(cost_saved, 2), "capped_wh_saved": round(wh_saved, 1),
    }


if __name__ == "__main__":
    run()
