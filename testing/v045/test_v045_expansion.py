"""Synthetic checks for the v0.4.5 stage-2 joint expansion (no data needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v045_stage2_expansion import Stage2Expander, make_expansion_hook  # noqa: E402

N_LIGHT, N_CH, N_TICKS = 2, 4, 300


def ti_fn(arr, t):
    out = np.zeros_like(arr, dtype=np.float64)
    s = int(round(t))
    if 0 <= s < arr.shape[1]:
        out[:, s:] = arr[:, : arr.shape[1] - s]
    return out


def build():
    """Two backbone chains (A near z=0, B near z=40) + candidates between/away."""
    pos, lab, ene = [], [], []

    def chain(z0, n, l, e_per):
        for i in range(n):
            pos.append([0.0, 0.0, z0 + i * 1.0]); lab.append(l); ene.append(e_per)

    chain(0, 10, 0, 20.0)      # backbone A  (label 0)
    chain(40, 10, 1, 20.0)     # backbone B  (label 1)
    chain(12, 3, 10, 2.0)      # small cand, 6 MeV, 3cm from A's end  -> A
    chain(30, 3, 11, 20.0)     # LARGE cand, 60 MeV -> must be skipped
    chain(37, 3, 12, 2.0)      # small cand, 6 MeV, near B            -> B
    chain(200, 3, 13, 2.0)     # small cand, disconnected             -> refused
    xyz = np.asarray(pos, float); labels = np.asarray(lab); e_pos = np.asarray(ene, float)
    hit_tpc = np.zeros(len(lab), int)          # everything in light TPC 0
    return xyz, labels, e_pos, hit_tpc


def make_img(amp, ch=(0, 1)):
    img = np.zeros((N_CH, N_TICKS))
    for c in ch:
        img[c, 0] = amp
        img[c, 1] = amp * 0.5
    return img


def main():
    xyz, labels, e_pos, hit_tpc = build()
    image_maps, cluster_to_tpcs = {}, {}
    for l in (0, 1, 10, 11, 12, 13):
        image_maps[(l, 0)] = make_img(1000.0 if l in (0, 1) else 400.0)
        cluster_to_tpcs[l] = [0]

    actual = np.zeros((N_LIGHT, N_CH, N_TICKS))
    # truth: A's light at t0=50 plus cand 10's light at t0=50 (so absorbing 10 helps)
    actual[0] += ti_fn(image_maps[(0, 0)], 50)
    actual[0] += ti_fn(image_maps[(10, 0)], 50)
    # B's light at t0=120 plus cand 12's light at 120
    actual[0] += ti_fn(image_maps[(1, 0)], 120)
    actual[0] += ti_fn(image_maps[(12, 0)], 120)

    base = np.zeros_like(actual)
    base[0] += ti_fn(image_maps[(0, 0)], 50)      # A already placed
    base[0] += ti_fn(image_maps[(1, 0)], 120)     # B already placed

    exp = Stage2Expander(
        xyz=xyz, e_pos=e_pos, labels=labels, image_maps=image_maps,
        cluster_to_tpcs=cluster_to_tpcs, actual=actual, base=base, ti_fn=ti_fn,
        backbone_labels=[0, 1], n_light=N_LIGHT, n_ch=N_CH, n_ticks=N_TICKS,
        verbose=True)

    print("candidate pool:", sorted(exp.cands), " (11 must be absent: 60 MeV > 30)")
    assert 11 not in exp.cands, "LARGE cluster entered the pool!"
    assert exp.stats["skipped_large"] == 1

    got = exp.expand({0: 50.0, 1: 120.0}, hit_tpc)
    print("absorbed:", {k: round(v, 1) for k, v in got.items()})
    assert 10 in got and abs(got[10] - 50.0) < 1e-6, "cand 10 should join family A @50"
    assert 12 in got and abs(got[12] - 120.0) < 1e-6, "cand 12 should join family B @120"
    assert 13 not in got, "disconnected candidate must be refused"
    assert 11 not in got

    # Voronoi: candidate 10 sits 3cm from A and ~25cm from B -> owned by A only.
    print("PASS: Voronoi ownership, 30 MeV veto, geodesic cut-off all behave")

    # hook composition: inner hook rejection must block expansion
    calls = {"n": 0}

    def inner_reject(lab, t0, baseA):
        calls["n"] += 1
        return False

    dec = {}
    hook = make_expansion_hook(inner_reject, exp, dec, hit_tpc, every=1)
    assert hook(0, 50.0, base) is False, "hook must propagate inner rejection"
    assert dec == {}, "rejected commit must not become a decided t0"
    print("PASS: rejected commits never trigger expansion")
    print("\nstats:", exp.stats)


if __name__ == "__main__":
    main()
