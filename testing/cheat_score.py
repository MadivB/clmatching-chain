"""Score the three-stage cheating trials: per-type efficiency + residuals."""
import glob, json, sys
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "v040cheat1_out"
rows = [json.loads(l) for fp in sorted(glob.glob(f"{d}/groups_worker*.jsonl")) for l in open(fp) if l.strip()]
print(f"{len(rows)} events from {d} (mode: {rows[0]['cheat']['mode']})")
print(f"{'type':<7}{'E truth[GeV]':>13}{'coverage':>10}{'eff(E)':>9}{'eff(hits)':>10}{'mean|res|':>10}{'med|res|':>9}")
for ty in (1, 2, 3):
    et = sum(r["cheat"][f"type{ty}"]["e_truth"] for r in rows)
    def wavg(key):
        num = sum(r["cheat"][f"type{ty}"][key] * r["cheat"][f"type{ty}"]["e_truth"]
                  for r in rows if r["cheat"][f"type{ty}"][key] is not None)
        return num / max(et, 1e-9)
    print(f"type {ty:<2}{et/1e3:>12.1f}{wavg('coverage'):>10.4f}{wavg('eff_e'):>9.4f}"
          f"{wavg('eff_n'):>10.4f}{wavg('mean_abs_res'):>9.1f}t{wavg('median_abs_res'):>8.1f}t")
