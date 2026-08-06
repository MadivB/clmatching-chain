"""All-hits correct-t0 statistics for the v0.3.0 D50 (both + force-coverage) run.

correct t0 := truth-valid hit whose assigned |ts_final - truth_t0| <= CUT ticks.
Reports count-weighted and ENERGY-weighted correct fraction, globally and split
by non-shower (tpc total E <= 1.4 GeV) vs shower (>1.4 GeV) TPCs.
"""
import glob, os
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030D50_out", "dumps")
CUT, SHOWER_E = 10.0, 1400.0

# accumulators: [N_total, N_truth, N_correct, E_total, E_truth, E_correct, N_assigned]
acc = {"all": np.zeros(7), "ns": np.zeros(7), "sh": np.zeros(7)}
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
for fp in files:
    d = np.load(fp, allow_pickle=True)
    tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    # per-tpc total energy -> shower mask (matches pipeline definition)
    tpc_e = np.zeros(int(tpc.max()) + 2)
    np.add.at(tpc_e, tpc, e)
    is_sh = tpc_e[tpc] > SHOWER_E
    truth_ok = np.isfinite(tr)
    asg = np.isfinite(tsf) & (tsf >= 0)
    correct = truth_ok & asg & (np.abs(tsf - tr) <= CUT)
    for name, sel in (("all", np.ones_like(is_sh)), ("ns", ~is_sh), ("sh", is_sh)):
        s = sel
        acc[name] += [s.sum(), (s & truth_ok).sum(), (s & correct).sum(),
                      e[s].sum(), e[s & truth_ok].sum(), e[s & correct].sum(),
                      (s & asg).sum()]

print(f"{len(files)} events | success = |reco-truth| <= {CUT:.0f} ticks (160 ns)\n")
hdr = f"{'cohort':>12} {'hits':>10} {'Etot(GeV)':>10} {'assigned%':>9} " \
      f"{'CORR/allHits':>13} {'CORR/truth':>11} {'Ecorr/Eall':>11} {'Ecorr/Etruth':>13}"
print(hdr); print("-" * len(hdr))
for name, lab in (("all", "ALL TPCs"), ("ns", "non-shower"), ("sh", "shower>1.4G")):
    n_tot, n_tr, n_cor, e_tot, e_tr, e_cor, n_asg = acc[name]
    print(f"{lab:>12} {n_tot:>10.0f} {e_tot/1e3:>10.1f} {100*n_asg/n_tot:>8.1f}% "
          f"{100*n_cor/max(n_tot,1):>12.2f}% {100*n_cor/max(n_tr,1):>10.2f}% "
          f"{100*e_cor/max(e_tot,1):>10.2f}% {100*e_cor/max(e_tr,1):>12.2f}%")

na = acc["all"]
print(f"\nHEADLINE (ALL hits, every TPC): "
      f"{100*na[2]/na[0]:.2f}% of hits correct | {100*na[5]/na[3]:.2f}% of energy correct")
print(f"(truth-valid hits = {100*na[1]/na[0]:.3f}% of all; the rest have no MC truth)")
