"""Export the sklearn HistGradientBoosting artifacts to model.json and a
data-coverage mask to coverage.json for browser-side inference.
Usage: python3 export_model.py [path/to/artifacts.joblib]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_DIR = Path(__file__).parent
DEFAULT = _DIR / "../modeling/artifacts.joblib"
CSV = _DIR / "../data_mod/housing_clean_nonrental.csv"


def export_tree(predictor):
    n = predictor.nodes
    return [[-1, 0.0, 0, 0, round(float(v), 5)] if leaf else
            [int(f), round(float(t), 6), int(l), int(r), 0.0]
            for v, f, t, l, r, leaf in zip(
                n["value"], n["feature_idx"], n["num_threshold"],
                n["left"], n["right"], n["is_leaf"])]


def export_model(m):
    return {"baseline": float(np.ravel(m._baseline_prediction)[0]),
            "trees": [export_tree(p[0]) for p in m._predictors]}


def main(path):
    M = joblib.load(path)
    out = {
        "features": ["lat", "lng", "log_area", "floor", "public",
                     "dist_cbd_km"],
        "qhat": float(M["qhat"]),
        "t_end": int(M["t_end"]),
        "models": {k: export_model(M[k]) for k in ("q10", "q50", "q90",
                                                   "trend")},
    }
    with open("model.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

    # verify: python re-implementation of the JS evaluator vs sklearn
    def js_predict(model, x):
        s = model["baseline"]
        for tree in model["trees"]:
            i = 0
            while tree[i][0] != -1:
                f, thr, l, r, _ = tree[i]
                i = l if x[f] <= thr else r
            s += tree[i][4]
        return s

    rng = np.random.default_rng(0)
    X = np.column_stack([
        rng.uniform(22.2, 22.5, 50), rng.uniform(113.9, 114.3, 50),
        np.log(rng.uniform(200, 1500, 50)), rng.integers(1, 50, 50),
        rng.integers(0, 2, 50),
        rng.uniform(0, 25, 50)])
    for k in ("q10", "q50", "q90"):
        ref = M[k].predict(
            __import__("pandas").DataFrame(X, columns=out["features"]))
        got = [js_predict(out["models"][k], x) for x in X]
        assert np.allclose(ref, got, atol=1e-3), k
    Xt = np.column_stack([X, np.full(50, M["t_end"])])
    cols = out["features"] + ["t"]
    ref = M["trend"].predict(__import__("pandas").DataFrame(Xt, columns=cols))
    got = [js_predict(out["models"]["trend"], x) for x in Xt]
    assert np.allclose(ref, got, atol=1e-3)
    print("verification OK; wrote model.json")


def coverage():
    """Grid mask of cells within one cell (~250 m) of a real transaction,
    so the heatmap never paints the sea or unbuilt terrain."""
    import pandas as pd

    lat0, lat1, lng0, lng1, step = 22.15, 22.58, 113.82, 114.45, 0.0025
    df = pd.read_csv(CSV, usecols=["lat", "lng"]).drop_duplicates()
    df = df[df.lat.between(lat0, lat1) & df.lng.between(lng0, lng1)]
    n_lat = int(round((lat1 - lat0) / step))
    n_lng = int(round((lng1 - lng0) / step))
    grid = np.zeros((n_lat, n_lng), dtype=bool)
    i = np.clip(((df.lat - lat0) / step).astype(int), 0, n_lat - 1)
    j = np.clip(((df.lng - lng0) / step).astype(int), 0, n_lng - 1)
    grid[i, j] = True
    dil = grid.copy()  # dilate by one cell in all 8 directions
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            dil |= np.roll(np.roll(grid, di, 0), dj, 1)
    with open("coverage.json", "w") as f:
        json.dump({"latMin": lat0, "lngMin": lng0, "step": step,
                   "nLat": n_lat, "nLng": n_lng,
                   "rows": ["".join("1" if c else "0" for c in row)
                            for row in dil]}, f, separators=(",", ":"))
    print(f"coverage.json: {int(dil.sum())}/{dil.size} cells covered")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    coverage()
