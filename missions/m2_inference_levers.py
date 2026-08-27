"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def reasoning_budget(rows: list, verbose: bool = True) -> dict:
    """Ext 4 (Your Turn #4, Guide.md §10): isolate $ and Wh spent on is_reasoning=1
    traffic vs is_reasoning=0, using the SAME optimized (cascade+cache+batch) pricing
    as the main run — reasoning requests still get cascade/cache/batch, they just
    also carry the ~80x energy multiplier from finops.sustainability.wh_per_query().
    """
    buckets = {True: {"n": 0, "tokens": 0, "cost": 0.0, "wh": 0.0},
               False: {"n": 0, "tokens": 0, "cost": 0.0, "wh": 0.0}}
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        tok = inp + out
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        wh = sustainability.wh_per_query(tok, is_reasoning=is_reasoning)
        b = buckets[is_reasoning]
        b["n"] += 1
        b["tokens"] += tok
        b["cost"] += cost
        b["wh"] += wh

    total_n = buckets[True]["n"] + buckets[False]["n"]
    total_cost = buckets[True]["cost"] + buckets[False]["cost"]
    total_wh = buckets[True]["wh"] + buckets[False]["wh"]
    r_n, nr_n = buckets[True]["n"], buckets[False]["n"]
    r_tok, nr_tok = buckets[True]["tokens"], buckets[False]["tokens"]
    r_cost, nr_cost = buckets[True]["cost"], buckets[False]["cost"]
    r_wh, nr_wh = buckets[True]["wh"], buckets[False]["wh"]

    result = {
        "reasoning_requests": r_n, "non_reasoning_requests": nr_n,
        "reasoning_tokens": r_tok, "non_reasoning_tokens": nr_tok,
        "reasoning_cost_daily": round(r_cost, 4), "non_reasoning_cost_daily": round(nr_cost, 4),
        "reasoning_wh_daily": round(r_wh, 2), "non_reasoning_wh_daily": round(nr_wh, 2),
        "reasoning_pct_of_cost": round(r_cost / total_cost * 100, 1) if total_cost else 0.0,
        "reasoning_pct_of_energy": round(r_wh / total_wh * 100, 1) if total_wh else 0.0,
        "reasoning_per_m": round(pricing.dollars_per_million(r_cost, r_tok), 3) if r_tok else 0.0,
        "non_reasoning_per_m": round(pricing.dollars_per_million(nr_cost, nr_tok), 3) if nr_tok else 0.0,
        "reasoning_wh_per_query": round(r_wh / r_n, 3) if r_n else 0.0,
        "non_reasoning_wh_per_query": round(nr_wh / nr_n, 3) if nr_n else 0.0,
    }

    if verbose:
        print("\n== Reasoning Budget (Ext 4) ==")
        print(f"requests : reasoning={r_n} ({r_n/total_n*100:.1f}%)   non-reasoning={nr_n} ({nr_n/total_n*100:.1f}%)")
        print(f"tokens   : reasoning={r_tok:,} ({r_tok/(r_tok+nr_tok)*100:.1f}%)   non-reasoning={nr_tok:,}")
        print(f"cost/day : reasoning=${r_cost:.3f} ({result['reasoning_pct_of_cost']:.1f}% of optimized spend)   non-reasoning=${nr_cost:.3f}")
        print(f"energy/day: reasoning={r_wh:.2f} Wh ({result['reasoning_pct_of_energy']:.1f}% of total)   non-reasoning={nr_wh:.2f} Wh")
        print(f"$/1M-token: reasoning=${result['reasoning_per_m']:.3f}   non-reasoning=${result['non_reasoning_per_m']:.3f}")
        print(f"Wh/query  : reasoning={result['reasoning_wh_per_query']:.3f}   non-reasoning={result['non_reasoning_wh_per_query']:.3f}"
              f"  (~{sustainability.REASONING_ENERGY_MULTIPLIER:.0f}x multiplier)")
        print("Routing rule: token_usage.csv has no confidence-score column, so this stays a policy"
              " proposal — gate the reasoning tier behind low fast-model confidence (< 0.6) or an"
              " explicit high-stakes task flag, and log confidence so the rule can be tuned on data.")

    return result


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

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

    reasoning = reasoning_budget(rows, verbose=verbose)

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_budget": reasoning,
    }


if __name__ == "__main__":
    run()
