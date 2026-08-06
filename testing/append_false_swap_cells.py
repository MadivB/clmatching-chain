"""Append a false-swap analysis section (v0.3 regressions) to swap_playground.ipynb."""
import json, os
HERE = os.path.abspath(os.path.dirname(__file__))
NB = os.path.join(HERE, "swap_playground.ipynb")
nb = json.load(open(NB))

def src(t):
    L = t.split("\n"); return [x + "\n" for x in L[:-1]] + [L[-1]]
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": src(t)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(t)}

# don't double-append
if any("False-swap analysis" in "".join(c["source"]) for c in nb["cells"]):
    print("section already present; skipping"); raise SystemExit

C = []
C.append(md(
"## 11 · False-swap analysis — where v0.3 REGRESSED (per-TPC)\n"
"\n"
"The per-TPC swap-fix is net-positive, but it *introduced* errors in a few TPCs. Cause: the per-TPC\n"
"chi2 was **fooled by a light deficit** — the ML predicts ~78% of the true amplitude, so an\n"
"under-predicted flash looks 'unfilled', and swapping a **bigger correct** cluster into it lowers\n"
"THIS TPC's chi2 (then cascades). These cells load the **pre-swap (D50)** and **post-swap (v0.3)**\n"
"states side by side so you can see exactly what broke and why. Everything here is per-TPC.\n"
"\n"
"Set `PICK_F` to 0/1/2 for the three regressions: TPC61 ev8, TPC26 ev5, TPC38 ev12."))

C.append(code(
"from M5p1 import visualizationTools as vt\n"
"from pulse_shapes import timeinterpolation\n"
"import glob as _glob\n"
"\n"
"FALSE = [('0000002', 8, 61), ('0000002', 5, 26), ('0000001', 12, 38)]   # (file, event, TPC)\n"
"PICK_F = 0\n"
"ffile, fev, ftpc = FALSE[PICK_F]\n"
"def _load(dumpdir, file, ev):\n"
"    return Event(np.load(os.path.join(HERE, dumpdir, 'dumps', f'{file}_ev{ev}.npz'), allow_pickle=True), file, ev)\n"
"PRE  = _load('v030D50fix_out', ffile, fev)     # before swap-fix\n"
"POST = _load('v030v3_out',     ffile, fev)     # after swap-fix (v0.3)\n"
"\n"
"# the swaps v0.3 applied IN THIS TPC (from the JSONL audit)\n"
"aud = []\n"
"for fp in _glob.glob(os.path.join(HERE, 'v030v3_out', 'groups_worker*.jsonl')):\n"
"    for l in open(fp):\n"
"        if not l.strip(): continue\n"
"        r = json.loads(l)\n"
"        if r['file'].split('.')[-3] == ffile and r['event'] == fev:\n"
"            aud = [a for a in ((r.get('phase3') or {}).get('swap_audit') or []) if a.get('tpc') == ftpc]\n"
"print(f'TPC{ftpc} ev{fev}: v0.3 applied {len(aud)} per-TPC swaps here')\n"
"for a in aud:\n"
"    print(f\"   L{a['A']}<->L{a['B']}  {a['tA']}<->{a['tB']}  chi2_ratio={a['chi2_ratio']}  truth {a.get('truthA')}/{a.get('truthB')}  good={a.get('good')}\")\n"
"\n"
"# per-cluster placement: pre -> post vs truth (who got broken)\n"
"print(f'\\nTPC{ftpc} cluster placements (pre -> post, truth):')\n"
"for L in sorted(set(int(x) for x in PRE.labels[(PRE.tpc == ftpc) & (PRE.labels >= 0)])):\n"
"    mp = (PRE.labels == L) & (PRE.tpc == ftpc)\n"
"    if mp.sum() < 20: continue\n"
"    ma = (POST.labels == L) & (POST.tpc == ftpc)\n"
"    trv = PRE.truth_t0[mp][np.isfinite(PRE.truth_t0[mp])]\n"
"    tp = np.median(PRE.ts_final[mp]); ta = (np.median(POST.ts_final[ma]) if ma.any() else np.nan)\n"
"    tr = (np.median(trv) if trv.size else np.nan)\n"
"    tag = 'ok' if abs(ta - tr) <= 10 else ('BROKEN (was correct!)' if abs(tp - tr) <= 10 else 'wrong')\n"
"    print(f'  L{L:<5} {tp:6.0f} -> {ta:6.0f}  (truth {tr:6.0f})  {tag}')"))

C.append(md(
"### Before / after — the deficit that fooled the chi2\n"
"Blue = actual light, orange = predicted. In BEFORE, look for a flash where orange sits well *below*\n"
"blue (a deficit); AFTER, the swap moved a bigger cluster there — lower chi2, but wrong truth."))

C.append(code(
"def _hits(ev):\n"
"    dt = np.dtype([('io_group','i4'),('x','f8'),('y','f8'),('z','f8'),('E','f8')])\n"
"    h = np.zeros(len(ev.x), dtype=dt); h['io_group'] = 2*ev.tpc + 1\n"
"    h['x'],h['y'],h['z'],h['E'] = ev.x, ev.y, ev.z, ev.e; return h\n"
"print('BEFORE (D50, pre swap-fix):')\n"
"vt.plot_tpc_light_and_charge(fev, ftpc, _hits(PRE), PRE.actual_light(ftpc), predicted=np.clip(PRE.predicted_light(ftpc),0,CLIP))\n"
"print('AFTER (v0.3, post swap-fix):')\n"
"vt.plot_tpc_light_and_charge(fev, ftpc, _hits(POST), POST.actual_light(ftpc), predicted=np.clip(POST.predicted_light(ftpc),0,CLIP))"))

C.append(md(
"### Why the chi2 accepted it — and whether the new spatial gate rejects it\n"
"The chi2 is amplitude-sensitive (fooled by the deficit). The **spatial cosine** is shape-normalized:\n"
"it asks whether each cluster's charge *position* matches the flash pattern, so a correctly-placed\n"
"cluster will NOT prefer the swap. `swap_fix` now requires BOTH a chi2 drop AND both parts to prefer\n"
"the swap spatially (`spatial_gate=True`). Here we re-derive both on the pre-swap state."))

C.append(code(
"def _tmpl(A, c):\n"
"    return POST.template(A, c) if POST.template(A, c) is not None else PRE.template(A, c)  # label-tpc invariant\n"
"def _flash(ev, c, t0, win=4):\n"
"    a = ev.actual_light(c); t = int(round(t0)) + PEAK\n"
"    col = a[:, max(0, t-win):min(a.shape[1], t+win+1)]; return col.sum(1), col.max(1) >= 0.98*CLIP\n"
"def _cos(u, v, m):\n"
"    u, v = u[~m], v[~m]; nu, nv = np.linalg.norm(u), np.linalg.norm(v)\n"
"    return float(u@v/(nu*nv)) if nu and nv else 0.0\n"
"for a in aud:\n"
"    A, B, tA, tB = a['A'], a['B'], a['tA'], a['tB']\n"
"    PA, PB = _tmpl(A, ftpc), _tmpl(B, ftpc)\n"
"    if PA is None or PB is None:\n"
"        print(f\"  L{A}<->L{B}: template not dumped\"); continue\n"
"    pa, pb = PA.sum(1), PB.sum(1)\n"
"    fA, sA = _flash(PRE, ftpc, tA); fB, sB = _flash(PRE, ftpc, tB)\n"
"    gA = _cos(pa, fB, sB) - _cos(pa, fA, sA)   # A prefers B's flash spatially?\n"
"    gB = _cos(pb, fA, sA) - _cos(pb, fB, sB)   # B prefers A's flash spatially?\n"
"    verdict = 'gate REJECTS -> fix works' if not (gA > 0 and gB > 0) else 'gate would STILL allow (needs more)'\n"
"    print(f\"  swap L{A}(@{tA:.0f})<->L{B}(@{tB:.0f}): chi2_ratio={a['chi2_ratio']} (accepted; lowers THIS TPC loss)\")\n"
"    print(f\"     spatial gA={gA:+.3f}  gB={gB:+.3f}   -> {verdict}\")"))

nb["cells"].extend(C)
json.dump(nb, open(NB, "w"), indent=1)
print(f"appended {len(C)} cells; notebook now {len(nb['cells'])} cells")
