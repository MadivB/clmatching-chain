"""Point swap_playground.ipynb at the v0.3 (post-swap-fix) dumps and refresh
TARGETS to the v0.3 remaining worst non-shower TPCs. Keeps the section-10 swap
demo loading the PRE-swap dumps (so it still shows a swap to fix)."""
import json, os
HERE = os.path.abspath(os.path.dirname(__file__))
NB = os.path.join(HERE, "swap_playground.ipynb")
V3 = "v030v3_out"          # post-swap-fix
PRE = "v030D50fix_out"     # pre-swap-fix (for the demo)

ctx = json.load(open(os.path.join(HERE, "bad_tpc_gallery", "_all_context.json")))
tg = []
for c in ctx:
    doms = c["dominant_wrong_clusters"][:2]
    tg.append({"i": c["idx"], "file": c["file"], "event": c["event"], "tpc": c["tpc"],
               "eff": c["eff"], "wrongE": c["wrong_E_real_cluster"],
               "swap_pair": [d["label"] for d in doms],
               "clusters": [{"L": d["label"], "E": d["cluster_E"], "placed": d["placed_t0"],
                             "true": d["true_t0"], "type": d["type"]} for d in doms]})

nb = json.load(open(NB))
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    # 1) point DUMPS at v0.3
    if "DUMPS = os.path.join(HERE," in src:
        new = []
        for l in cell["source"]:
            if l.strip().startswith("DUMPS = os.path.join(HERE,"):
                new.append(f"DUMPS = os.path.join(HERE, '{V3}', 'dumps')   # v0.3 (D50 + per-TPC swap-fix)\n")
            elif "if not os.path.isdir(DUMPS)" in l:
                new.append(f"if not os.path.isdir(DUMPS): DUMPS = os.path.join(HERE, '{PRE}', 'dumps')\n")
            else:
                new.append(l)
        cell["source"] = new
    # 2) refresh TARGETS to v0.3 remaining
    if src.strip().startswith("TARGETS = "):
        after = src.split("\nprint(", 1)
        tail = "\nprint(" + after[1] if len(after) > 1 else ""
        cell["source"] = [l + "\n" for l in ("TARGETS = " + json.dumps(tg, indent=1)).split("\n")]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        for l in tail.split("\n"):
            cell["source"].append("\n" + l if l else "\n")
    # 3) keep the section-10 demo on the PRE-swap dumps
    if "EV2 = load_event('0000001', 11)" in src:
        cell["source"] = [l.replace(
            "EV2 = load_event('0000001', 11)",
            f"EV2 = Event(np.load(os.path.join(HERE,'{PRE}','dumps','0000001_ev11.npz'),allow_pickle=True),'0000001',11)  # PRE-swap, to demo a swap")
            for l in cell["source"]]

json.dump(nb, open(NB, "w"), indent=1)
print(f"notebook -> DUMPS={V3}, {len(tg)} v0.3 TARGETS, demo on {PRE}")
