"""Headless end-to-end smoke test of phase1_rescue_testing.ipynb (run on a GPU node).

Executes every code cell in order, then re-runs the RESET -> trigger -> rescue -> adopt
chain a second time and asserts the rescued accuracy reproduces exactly (validates the
notebook's re-tuning loop), then exercises the WRONG_STAGE="phase1" branch.
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
try:
    import plotly.graph_objects as go
    go.Figure.show = lambda self, *a, **k: None
except Exception:
    pass

HERE = os.path.abspath(os.path.dirname(__file__))
os.chdir(HERE)
nb = json.load(open(os.path.join(HERE, "phase1_rescue_testing.ipynb")))
cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
g = {"__name__": "__main__"}


def find(marker):
    return next(i for i, s in cells if marker in s)


def run(i, src=None, tag=""):
    if src is None:
        src = dict(cells)[i]
    print(f"\n================ CELL {i} {tag}================", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), g)


for i, src in cells:
    run(i, src)

by_idx = dict(cells)
CH_RESET = find("pristine phase-1 state captured")
CH_TRIG = find("trigger constants")
CH_A = find("all_stay_groups = []")
CH_B = find("detaches queued")
CH_N1 = find("n1_rem_moves = {}")
CH_ADOPT = find("_ADOPTED = True")
CH_SWAP = find("SWAP_SWEEPS")
CH_PICK = find('WRONG_STAGE   = "rescued"')
CH_FIG = find("EXAMPLE_RANK")
CHAIN = (CH_RESET, CH_TRIG, CH_A, CH_B, CH_N1, CH_ADOPT, CH_SWAP)

print("\n######## RESET RE-RUN (tunability loop) ########", flush=True)
acc_first = g["acc_sw"]
for i in CHAIN:
    run(i, tag="(rerun) ")
assert abs(g["acc_sw"] - acc_first) < 1e-12, f"reset loop not reproducible: {acc_first} vs {g['acc_sw']}"
print(f"\nRESET loop reproducible: acc_sw = {g['acc_sw']:.6f} both times")

print("\n######## WRONG_STAGE='phase1' branch ########", flush=True)
run(CH_PICK, by_idx[CH_PICK].replace('WRONG_STAGE   = "rescued"', 'WRONG_STAGE   = "phase1"'),
    "(phase1 stage) ")
run(CH_FIG, by_idx[CH_FIG].replace("FIG_STAGE = WRONG_STAGE", 'FIG_STAGE = "phase1"'),
    "(phase1 stage) ")

print("SMOKE TEST PASSED", flush=True)
