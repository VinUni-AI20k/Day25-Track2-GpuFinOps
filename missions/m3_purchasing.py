"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

    ext = _extension1_tier_policy(jobs, cat, on_demand_monthly, optimized_monthly, verbose=verbose)

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "extension1_v2_monthly": ext["v2_monthly"], "extension1_v2_savings_pct": ext["v2_savings_pct"]}


def _extension1_tier_policy(jobs, cat, on_demand_monthly, v1_monthly, verbose: bool = True) -> dict:
    """Your Turn Extension 1 — recommend_tier() with gpu_type interrupt rate + job_days
    (1yr vs 3yr reserved) factored in. Compares against the base policy above."""
    v2_monthly = 0.0
    v2_recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])

        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=days)
        if tier == "spot":
            opt_cost = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
        elif tier == "reserved_3yr":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        elif tier == "reserved_1yr":
            opt_cost = gpu_hours * num(c["reserved_1yr_hr"])
        else:
            opt_cost = gpu_hours * od
        v2_monthly += opt_cost
        v2_recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier, "optimized": round(opt_cost)})

    v2_savings_pct = (on_demand_monthly - v2_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0
    if verbose:
        print("\n-- Extension 1: recommend_tier() w/ GPU interrupt-rate + 1yr-vs-3yr job_days --")
        print(f"{'job':18}{'gpu':7}{'tier':13}{'optimized':>12}")
        for r in v2_recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:13}${r['optimized']:>11,}")
        print(f"monthly optimized: base policy ${v1_monthly:,.0f}"
              f" -> v2 ${v2_monthly:,.0f}  ({v2_savings_pct:.1f}% saved vs on-demand, "
              f"was {(on_demand_monthly - v1_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0:.1f}%)")
    if verbose:
        _extension1_matrix_demo()
    return {"v2_monthly": round(v2_monthly), "v2_savings_pct": round(v2_savings_pct, 1), "recommendations": v2_recs}


def _extension1_matrix_demo() -> None:
    """This fleet's 8 workloads don't happen to cross the new interrupt-rate/duration
    thresholds (H100 dominates the interruptible jobs, and every reserved-eligible job
    runs 30/30 days) so v1 and v2 land on the same $. Show the matrix directly against
    recommend_tier() so the new factors are visibly exercised on other fleet shapes."""
    print("\n-- Extension 1: tier matrix (gpu_type x hours/day x interruptible), job_days=30 --")
    print(f"{'gpu':7}{'hours/day':11}{'base (no gpu_type)':22}{'v2 (gpu_type+job_days)':22}")
    for gtype in ["H100", "A100", "A10G", "L4"]:
        for hpd in (4, 12, 20):
            for interruptible in (True, False):
                base = pricing.recommend_tier(hpd, interruptible)
                v2 = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=30)
                flag = "  <- differs" if base != v2 and v2 not in ("reserved",) else ""
                label = f"{hpd}h/interrupt={interruptible}"
                print(f"{gtype:7}{label:23}{base:22}{v2}{flag}")


if __name__ == "__main__":
    run()
