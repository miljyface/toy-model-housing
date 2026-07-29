"""Export the sklearn HistGradientBoosting artifacts to model.json for
browser-side inference. Usage: python3 export_model.py [path/to/artifacts.joblib]
"""
import json
import sys

import joblib
import numpy as np

DEFAULT = ("/Users/guanrong/Documents/Uchicago_Summer_Datsci/Project/"
           "modeling/artifacts.joblib")


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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
