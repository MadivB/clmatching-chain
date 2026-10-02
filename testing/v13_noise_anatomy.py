"""Disintegrate the 'unclustered (noise) hits' class of the assignment ledger.
For every label -1 hit with truth: distance to the nearest clustered hit, size of its own noise group (single linkage,
eps 4 cm = the frontend's leftover DBSCAN radius), whether a reco family exists within 10 ticks of its truth time in its
TPC, whether the nearest clustered hit belongs to the same interaction (same truth time) and whether it is correct.
python v13_noise_anatomy.py <arm_dir>"""
import glob
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CUT = 10.0; EPS = 4.0
D_BINS = [0, 1, 2, 4, 10, 25, 1e9]; D_LAB = ["< 1 cm (touching a cluster)", "1-2 cm", "2-4 cm", "4-10 cm", "10-25 cm", "> 25 cm"]
G_BINS = [1, 2, 3, 5, 10, 30, 1e9]; G_LAB = ["1 hit (alone)", "2 hits", "3-4", "5-9", "10-29", ">= 30"]


def main(arm):
    A = np.zeros((len(D_LAB), len(G_LAB), 6))   # E, E_correct, E_undiscovered, E_neighbour_same_interaction, E_neighbour_correct, E_assigned_neighbour_time
    tot_noise_e = 0.0; n_hits = 0; e_nan = 0.0
    for ff in sorted(glob.glob(f"{arm}/full_dumps/*.npz")):
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885); asg = np.isfinite(ts) & (ts >= 0)
        good = ok & asg & (np.abs(ts - tru) <= CUT)
        noise = labs < 0; cl = labs >= 0
        # nearest clustered hit
        tree_c = cKDTree(xyz[cl]); d_c, j_c = tree_c.query(xyz[noise], k=1)
        idx_c = np.flatnonzero(cl)[j_c]
        # noise groups (single linkage within EPS among noise hits)
        pn = np.flatnonzero(noise); tree_n = cKDTree(xyz[pn]); pairs = tree_n.query_pairs(EPS, output_type="ndarray")
        n_n = pn.size
        if pairs.size:
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_n, n_n))
            _, comp = connected_components(g, directed=False)
        else:
            comp = np.arange(n_n)
        gsize = np.bincount(comp)[comp]
        # reco families per TPC: rounded final times of assigned clustered hits
        fam = {}
        for t_ in np.unique(tpc):
            m = cl & asg & (tpc == t_)
            fam[int(t_)] = np.unique(np.round(ts[m])) if m.any() else np.array([])
        e_n = e[pn]; ok_n = ok[pn]; good_n = good[pn]; tru_n = tru[pn]; ts_n = ts[pn]; tpc_n = tpc[pn]
        und = np.array([not (np.abs(fam.get(int(t_), np.array([])) - tt).min() <= CUT) if fam.get(int(t_), np.array([])).size else True for t_, tt in zip(tpc_n, tru_n)])
        same = np.isfinite(tru[idx_c]) & (np.abs(tru[idx_c] - tru_n) <= 2.0)
        nb_good = good[idx_c]
        at_nb = np.isfinite(ts_n) & np.isfinite(ts[idx_c]) & (np.abs(ts_n - ts[idx_c]) < 0.5)
        di = np.clip(np.searchsorted(D_BINS, d_c, side="right") - 1, 0, len(D_LAB) - 1)
        gi = np.clip(np.searchsorted(G_BINS, gsize, side="right") - 1, 0, len(G_LAB) - 1)
        for a in range(len(D_LAB)):
            for b in range(len(G_LAB)):
                m = ok_n & (di == a) & (gi == b)
                if m.any():
                    A[a, b] += [e_n[m].sum(), e_n[m & good_n].sum(), e_n[m & und].sum(), e_n[m & same].sum(), e_n[m & nb_good].sum(), e_n[m & at_nb].sum()]
        tot_noise_e += e_n[ok_n].sum(); n_hits += int(ok_n.sum()); e_nan += e_n[ok_n & ~np.isfinite(ts_n)].sum()
    print(f"noise (label -1) hits with truth: {n_hits} hits, {tot_noise_e/1e3:.2f} GeV; energy left without a time (isolated > 25 cm): {e_nan/1e3:.2f} GeV")
    print("\nBy isolation (distance to the nearest clustered hit): energy GeV | correct | truth flash not discovered in the TPC | nearest clustered hit is the same interaction | given the neighbour's time | neighbour itself correct")
    for a in range(len(D_LAB)):
        r = A[a].sum(0)
        if r[0] > 0:
            print(f"  {D_LAB[a]:30s} {r[0]/1e3:6.2f} ({100*r[0]/tot_noise_e:4.1f} %) | {r[1]/r[0]:.3f} | {r[2]/r[0]:.3f} | {r[3]/r[0]:.3f} | {r[5]/r[0]:.3f} | {r[4]/r[0]:.3f}")
    print("\nBy noise-group size (single linkage, 4 cm):")
    for b in range(len(G_LAB)):
        r = A[:, b].sum(0)
        if r[0] > 0:
            print(f"  {G_LAB[b]:16s} {r[0]/1e3:6.2f} GeV ({100*r[0]/tot_noise_e:4.1f} %) | correct {r[1]/r[0]:.3f} | undiscovered {r[2]/r[0]:.3f} | neighbour same interaction {r[3]/r[0]:.3f}")
    print("\nCross table: rows = isolation, columns = group size; cell = energy GeV / correct fraction")
    print(" " * 30 + "".join(f"{g:>16s}" for g in G_LAB))
    for a in range(len(D_LAB)):
        print(f"  {D_LAB[a]:28s}" + "".join(f"{A[a,b,0]/1e3:7.2f}/{A[a,b,1]/max(A[a,b,0],1e-9):.2f}    " if A[a, b, 0] > 0 else " " * 16 for b in range(len(G_LAB))))
    # the two populations the user asked about
    alone = A[:, :2].sum((0, 1)); grouped = A[:, 2:].sum((0, 1)); touching = A[0].sum(0); far = A[3:].sum((0, 1))
    print(f"\nSUMMARY: alone (1-2 hits within 4 cm): {alone[0]/1e3:.2f} GeV, correct {alone[1]/alone[0]:.3f}; grouped (>= 3 noise hits within 4 cm): {grouped[0]/1e3:.2f} GeV, correct {grouped[1]/grouped[0]:.3f}")
    print(f"         touching a cluster (< 1 cm): {touching[0]/1e3:.2f} GeV, correct {touching[1]/touching[0]:.3f}, neighbour same interaction {touching[3]/touching[0]:.3f}; >= 4 cm from any cluster: {far[0]/1e3:.2f} GeV, correct {far[1]/far[0]:.3f}, undiscovered {far[2]/far[0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
