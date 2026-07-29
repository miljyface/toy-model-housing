# HK Housing Price Model — Web Demo

Static, client-side demo of a Hong Kong housing price model. Click anywhere on the
map, enter floor, saleable area (ft²), and whether the property is public housing;
the page returns an 80% price interval, a median point estimate, and the trailing
12-month local market trend.

Everything runs in the browser: the gradient-boosted quantile models
(scikit-learn `HistGradientBoostingRegressor`, conformally calibrated) are exported
to `model.json` and evaluated by a ~15-line tree walker in `index.html`.
No backend, no build step.

## Files

- `index.html` — map UI (Leaflet + OpenStreetMap) and inference code
- `model.json` — exported trees for the q10/q50/q90 price quantiles and the
  time-trend model, plus the conformal correction `qhat`
- `export_model.py` — regenerates `model.json` from the training artifact and
  verifies the exported trees reproduce sklearn's predictions

## Run locally

```
python3 -m http.server
# open http://localhost:8000
```

## Deploy to GitHub Pages

```
gh repo create hk-housing-price-demo --public --source . --push   # or push manually
```

Then: repository Settings → Pages → Source: `main` branch, `/ (root)` → Save.
The site appears at `https://<user>.github.io/hk-housing-price-demo/`.

## Model notes

Trained on ~48k HK transactions (2020–2023). Features derived from the click +
form inputs only: lat, lng, log(area), floor, public flag, distance to Central.
The interval is a CQR-calibrated [q10, q90] band (~86% observed coverage on a
2023 temporal holdout); the trend is the model's trailing 12-month appreciation
at that location, evaluated at the end of the training window (March 2023) —
it is a historical market trend, not a forward forecast.
