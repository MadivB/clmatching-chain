"""Precompute per-sample variance maps with the trained variance predictor (the chain's
var_prediction checkpoint) for every dump in a dump dir, so the offline v2.0 solver can use
w = 1/var instead of the Poisson form (V200_VARDIR).
python v200_varcache.py <dump_dir> [--out <dir>]   (GPU)"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", "CLMatching_AlphaRelease"))
for p in (os.path.join(BASE, "NewMLSection"), BASE):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402
from var_prediction.inference_ndfl import load_model, predict  # noqa: E402

VAR_CKPT = ("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/var_prediction/"
            "runs/var_run_perceiver_aligned_0001000_v2/best_model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dump_dir, "var_cache")
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_model(VAR_CKPT, device=dev, waveform_len=1000, num_channels=120)
    print("variance model on", dev, {k: meta[k] for k in list(meta)[:6]} if isinstance(meta, dict) else "", flush=True)
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.dump_dir, "full_dumps", "*.npz")))
    for f in files:
        tag = os.path.basename(f)[:-4]
        dst = os.path.join(out, tag + ".npy")
        if os.path.exists(dst):
            continue
        z = np.load(f, allow_pickle=True)
        A = np.asarray(z["full_light"], np.float32)              # (n_light, 120, 1000)
        var = predict(model, A, batch_size=8, input_scale=1e-3, target_scale=1e-3, device=dev,
                      return_variance=True, min_sigma=1.0).astype(np.float32)
        var = np.maximum(var, 1.0)
        np.save(dst, var)
        pois = 200.0 * np.maximum(A, 100.0)
        m = A > 1000
        print(f"{tag}: var/poisson median {np.median(var[m] / pois[m]):.2f} (A>1000), {np.median(var[~m] / pois[~m]):.2f} (A<=1000)  {time.time()-t0:.0f}s", flush=True)
    print("done", len(files), "events ->", out)


if __name__ == "__main__":
    main()
