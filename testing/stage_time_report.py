"""Per-stage time-consumption report for a pipeline output dir.

Usage: python stage_time_report.py <out_dir> [baseline_dir]
Prints the full stage table (chain stages + stage-1 internals + stage-6
internals + tail accounting), optionally with a baseline column.
"""
import glob
import json
import sys

import numpy as np


def load(d):
    return [json.loads(l) for f in glob.glob(f"{d}/groups_worker*.jsonl") for l in open(f)]


def agg(rows, getter):
    out = {}
    for r in rows:
        for k, v in (getter(r) or {}).items():
            if isinstance(v, (int, float)):
                out.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in out.items()}


def report(rows, base=None):
    tev = float(np.mean([r["t_event_s"] for r in rows]))
    st = agg(rows, lambda r: r.get("stage_t"))
    s1 = agg(rows, lambda r: r.get("stage1_t"))
    s6 = agg(rows, lambda r: (r["phase3"].get("subtick") or {}))
    le = float(np.mean([(r["phase3"].get("le_retime") or {}).get("time_s", 0) for r in rows]))
    bst = agg(base, lambda r: r.get("stage_t")) if base else {}
    btev = float(np.mean([r["t_event_s"] for r in base])) if base else None

    print(f"{'stage':30s}{'s/ev':>7s}{'share':>7s}" + ("{:>9s}".format("baseline") if base else ""))
    order = sorted(st, key=lambda k: -st[k])
    for k in order:
        line = f"{k:30s}{st[k]:7.1f}{100*st[k]/tev:6.1f}%"
        if base:
            line += f"{bst.get(k, 0):9.1f}"
        print(line)
        if k == "1_clustering_prediction" and s1:
            for kk in ("prediction_s", "clustering_s", "track_stage_s", "load_event_s", "flash_table_s"):
                if kk in s1:
                    print(f"    - {kk:24s}{s1[kk]:7.1f}")
    t6 = s6.get("time_s", 0.0)
    print(f"{'6_subtick':30s}{t6:7.1f}{100*t6/tev:6.1f}%")
    for kk in ("t_vox", "t_model", "t_render", "t_grid", "t_load"):
        if kk in s6 and s6[kk] > 0.005:
            print(f"    - {kk:24s}{s6[kk]:7.1f}")
    print(f"{'5_le_retime':30s}{le:7.1f}{100*le/tev:6.1f}%")
    acct = sum(st.values()) + t6 + le
    print(f"{'io/metrics/other':30s}{tev-acct:7.1f}{100*(tev-acct)/tev:6.1f}%")
    tail = f"   (baseline {btev:.1f} -> {btev/2:.1f}/GPU)" if btev else ""
    print(f"{'TOTAL':30s}{tev:7.1f}   = {tev/2:.1f} s/ev/GPU{tail}")


def main():
    rows = load(sys.argv[1])
    base = load(sys.argv[2]) if len(sys.argv) > 2 else None
    report(rows, base)


if __name__ == "__main__":
    main()
