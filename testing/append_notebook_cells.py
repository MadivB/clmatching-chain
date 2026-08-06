"""Append per-TPC pairwise-swap cells to swap_playground.ipynb (preserve existing
cells) + point DUMPS at the fixed dumps."""
import json, os
HERE = os.path.abspath(os.path.dirname(__file__))
NB = os.path.join(HERE, "swap_playground.ipynb")
nb = json.load(open(NB))

def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": src(t)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(t)}

# --- 1) fix DUMPS in cell 2 (imports) to prefer the FIXED dumps ---
for c in nb["cells"]:
    if c["cell_type"] == "code" and any("DUMPS = os.path.join(HERE, 'v030D50_out'" in l for l in c["source"]):
        new = []
        for l in c["source"]:
            if "DUMPS = os.path.join(HERE, 'v030D50_out'" in l:
                new.append("DUMPS = os.path.join(HERE, 'v030D50fix_out', 'dumps')   # FIXED base_image (correct predicted light)\n")
                new.append("if not os.path.isdir(DUMPS): DUMPS = os.path.join(HERE, 'v030D50_out', 'dumps')\n")
            else:
                new.append(l)
        c["source"] = new
        break

# --- 2) append the per-TPC swap section ---
CELLS = []
CELLS.append(md(
"## 10 · Per-TPC pairwise swap (the RIGHT unit)\n"
"\n"
"We **cannot** trust the multi-TPC clustering — one label can hold hits from two different\n"
"interactions, and the pipeline is allowed to split a cluster's hits across TPCs. So the swap must\n"
"act on **cluster-PARTS inside a single TPC**, and it must be a **joint pairwise** swap: moving one\n"
"part alone leaves a light deficit (chi2 gets worse), but swapping *both* parts wins.\n"
"\n"
"Worked example: `0000001` ev11, **TPC 43** — L95 (reco 482, truth 160) and L86 (reco 160, truth 481)\n"
"are swapped. Both the per-channel spatial cosine and the support-masked chi2 detect it."))

CELLS.append(code(
"# per-TPC swap detector: cluster-PARTS in one TPC, joint pairwise swap\n"
"from pulse_shapes import timeinterpolation\n"
"\n"
"EV2 = load_event('0000001', 11)     # the L95<->L86 multi-TPC example\n"
"SWAP_TPC = 43\n"
"\n"
"def part_t0(ev, L, c):\n"
"    m = (ev.labels == L) & (ev.tpc == c) & np.isfinite(ev.ts_final)\n"
"    return float(np.median(ev.ts_final[m])) if m.any() else None\n"
"def part_truth(ev, L, c):\n"
"    m = (ev.labels == L) & (ev.tpc == c) & np.isfinite(ev.truth_t0)\n"
"    return float(np.median(ev.truth_t0[m])) if m.any() else None\n"
"def _flash_prof(ev, c, t0, win=4):\n"
"    a = ev.actual_light(c); t = int(round(t0)) + PEAK\n"
"    col = a[:, max(0, t-win):min(a.shape[1], t+win+1)]\n"
"    return col.sum(1), col.max(1) >= 0.98*CLIP\n"
"def _cos(u, v, m):\n"
"    u, v = u[~m], v[~m]; nu, nv = np.linalg.norm(u), np.linalg.norm(v)\n"
"    return float(u@v/(nu*nv)) if nu and nv else 0.0\n"
"\n"
"def per_tpc_chi2(ev, c, A, B, tA, tB, aAt, aBt):\n"
"    \"\"\"support-masked per-TPC light chi2 with A placed at aAt and B at aBt.\"\"\"\n"
"    PA, PB = ev.template(A, c), ev.template(B, c)\n"
"    bw = ev.predicted_light(c) - timeinterpolation(PA, shift=tA, baseline=0) - timeinterpolation(PB, shift=tB, baseline=0)\n"
"    pred = np.clip(bw + timeinterpolation(PA, shift=aAt, baseline=0) + timeinterpolation(PB, shift=aBt, baseline=0), 0, CLIP)\n"
"    act = ev.actual_light(c)\n"
"    supp = (PA.sum(1) > 0.02*PA.sum(1).max()) | (PB.sum(1) > 0.02*PB.sum(1).max())\n"
"    lo, hi = max(0, int(min(tA, tB))+PEAK-15), min(act.shape[1], int(max(tA, tB))+PEAK+50)\n"
"    a = act[np.ix_(supp, np.arange(lo, hi))]; pr = np.minimum(pred[np.ix_(supp, np.arange(lo, hi))], CLIP)\n"
"    v = GAIN*np.maximum(a, 100.0); m = a < 0.98*CLIP\n"
"    return float((((pr-a)**2/v)[m]).sum())\n"
"\n"
"def find_tpc_swaps(ev, c, min_hits=20, min_sep=20):\n"
"    \"\"\"pairs of templated cluster-parts in TPC c whose JOINT swap lowers chi2 AND\n"
"    both parts prefer the exchange by spatial cosine.\"\"\"\n"
"    parts = [int(L) for L in np.unique(ev.labels[(ev.tpc == c) & (ev.labels >= 0)])\n"
"             if ((ev.labels == L) & (ev.tpc == c)).sum() >= min_hits and ev.template(L, c) is not None]\n"
"    out = []\n"
"    for i in range(len(parts)):\n"
"        for j in range(i+1, len(parts)):\n"
"            A, B = parts[i], parts[j]; tA, tB = part_t0(ev, A, c), part_t0(ev, B, c)\n"
"            if abs(tA - tB) < min_sep: continue\n"
"            fA, sA = _flash_prof(ev, c, tA); fB, sB = _flash_prof(ev, c, tB)\n"
"            gA = _cos(ev.template(A, c).sum(1), fB, sB) - _cos(ev.template(A, c).sum(1), fA, sA)\n"
"            gB = _cos(ev.template(B, c).sum(1), fA, sA) - _cos(ev.template(B, c).sum(1), fB, sB)\n"
"            cur = per_tpc_chi2(ev, c, A, B, tA, tB, tA, tB)\n"
"            swp = per_tpc_chi2(ev, c, A, B, tA, tB, tB, tA)\n"
"            out.append(dict(A=A, B=B, tA=tA, tB=tB, gA=gA, gB=gB, ratio=swp/cur,\n"
"                            mutual=(gA > 0 and gB > 0), chi2_better=(swp < cur),\n"
"                            truthA=part_truth(ev, A, c), truthB=part_truth(ev, B, c)))\n"
"    return sorted(out, key=lambda r: r['ratio'])\n"
"\n"
"sw = find_tpc_swaps(EV2, SWAP_TPC)\n"
"print(f\"per-TPC swap candidates in {EV2.file} ev{EV2.event} TPC{SWAP_TPC}:\")\n"
"print(f\"{'A':>5}{'B':>5}{'tA':>6}{'tB':>6}{'truthA':>7}{'truthB':>7}{'cosgA':>7}{'cosgB':>7}{'chi2ratio':>10}  verdict\")\n"
"for r in sw:\n"
"    v = 'SWAP' if (r['mutual'] and r['chi2_better']) else ('chi2-only' if r['chi2_better'] else '-')\n"
"    print(f\"{r['A']:>5}{r['B']:>5}{r['tA']:>6.0f}{r['tB']:>6.0f}{r['truthA']:>7.0f}{r['truthB']:>7.0f}\"\n"
"          f\"{r['gA']:>+7.3f}{r['gB']:>+7.3f}{r['ratio']:>10.3f}  {v}\")"))

CELLS.append(md(
"### Before / after per-TPC swap displays\n"
"Applies the top swap candidate in `SWAP_TPC` and shows the TPC light+charge **before** and **after**.\n"
"After the swap the predicted (orange) fills the deficit at the vacated flash and stops over-filling\n"
"the other — the light budget is restored."))

CELLS.append(code(
"# BEFORE / AFTER per-TPC swap (two figures)\n"
"from M5p1 import visualizationTools as vt\n"
"best = [r for r in sw if r['mutual'] and r['chi2_better']]\n"
"assert best, 'no confident per-TPC swap in SWAP_TPC; try another TPC'\n"
"r = best[0]; A, B, tA, tB, c = r['A'], r['B'], r['tA'], r['tB'], SWAP_TPC\n"
"PA, PB = EV2.template(A, c), EV2.template(B, c)\n"
"act = EV2.actual_light(c); base = EV2.predicted_light(c)\n"
"pred_before = np.clip(base, 0, CLIP)\n"
"pred_after = np.clip(base - timeinterpolation(PA, shift=tA, baseline=0) - timeinterpolation(PB, shift=tB, baseline=0)\n"
"                          + timeinterpolation(PA, shift=tB, baseline=0) + timeinterpolation(PB, shift=tA, baseline=0), 0, CLIP)\n"
"_dt = np.dtype([('io_group', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('E', 'f8')])\n"
"hf = np.zeros(len(EV2.x), dtype=_dt)\n"
"hf['io_group'] = 2*EV2.tpc + 1; hf['x'], hf['y'], hf['z'], hf['E'] = EV2.x, EV2.y, EV2.z, EV2.e\n"
"print(f'===== BEFORE swap:  L{A}@{tA:.0f}, L{B}@{tB:.0f}   (truth {r[\"truthA\"]:.0f}/{r[\"truthB\"]:.0f}) =====')\n"
"vt.plot_tpc_light_and_charge(EV2.event, c, hf, act, predicted=pred_before)\n"
"print(f'===== AFTER swap:   L{A}@{tB:.0f}, L{B}@{tA:.0f}   (now matches truth) =====')\n"
"vt.plot_tpc_light_and_charge(EV2.event, c, hf, act, predicted=pred_after)"))

nb["cells"].extend(CELLS)
json.dump(nb, open(NB, "w"), indent=1)
print(f"appended {len(CELLS)} cells; notebook now has {len(nb['cells'])} cells; DUMPS -> fixed")
