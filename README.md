# E-commerce Promotion Evaluation System

A graduation project built with Flask, Pandas, scikit-learn, and XGBoost. This repository combines **offline data processing and modelling scripts, a runnable Flask API, and plain HTML/CSS/JavaScript pages** for campaign comparison, user behaviour analysis, and prediction demonstrations.

## What I built

- **Campaign evaluation:** Group-level analysis of weekly store sales in the WA dataset, propensity score matching (PSM), and difference-in-differences (DiD). The dashboard presents campaign averages, weekly trends, and effect comparisons.
- **User behaviour:** User-level RFM, browse/favourite, active-time, and category features aggregated from event logs. An XGBoost conversion model produces scores and feature-importance outputs for segmentation, funnel, and prediction views.
- **Incremental-response analysis:** Treatment/control groups aggregated from promotion exposure logs; S-Learner and T-Learner estimate response differences and export Uplift segments.
- **Application integration:** Flask loads derived CSV files and a serialized model at startup, exposes JSON APIs, gates analysis and prediction endpoints by session roles, and serves browser dashboards rendered with ECharts.

```text
Raw datasets (not distributed here)
  -> scripts/ offline cleaning, feature engineering, statistics, training
  -> backend/data/ derived demo CSV files + serialized model
  -> backend/app.py startup loading, calculations, lookups, JSON APIs
  -> backend/static/ browser pages and ECharts visualisations
```

## Code review guide

| Stage | Source | Input, output, and implementation |
| --- | --- | --- |
| Store campaign analysis | [`wa_analysis.py`](scripts/01_WA分析/wa_analysis.py) | Aggregates store sales by campaign and produces averages, ANOVA/regression results, and plots. |
| PSM | [`wa_psm.py`](scripts/02_PSM因果推断/wa_psm.py) | Fits store-size/age propensity scores for campaigns 1 and 2; applies common-support filtering, nearest-neighbour matching, and covariate-balance checks. |
| DiD | [`wa_did.py`](scripts/01_WA分析/wa_did.py) | Treats weeks 1–2 / 3–4 as pre/post periods, compares campaigns 1 and 2 by difference-in-differences, and exports regression results and plots. |
| User features | [`ub_feature_engineering.py`](scripts/04_UserBehavior分析/ub_feature_engineering.py) | Aggregates event logs into user-level RFM, browse/favourite counts, active days, and time/category features; writes `user_features.csv`. |
| Conversion model | [`ub_xgboost_model.py`](scripts/04_UserBehavior分析/ub_xgboost_model.py) | Stratified train/test split, XGBoost training, AUC/KS/Lift and feature importance; exports scoring CSV and a `joblib` model. |
| Exposure Uplift | [`log_uplift_model.py`](scripts/05_日志数据分析/log_uplift_model.py) | Aggregates exposure and conversion logs, fits S-/T-Learners, and outputs per-user `uplift_t` and segment CSV files. |
| Flask service | [`backend/app.py`](backend/app.py) | Preloads data/model; APIs calculate KPIs, trends, PSM/DiD, RFM and segments, as well as prediction and batch-upload responses. |
| Browser UI | [`backend/static/`](backend/static/) | Plain pages call the HTTP/JSON APIs and use ECharts for trends, funnels, ROC plots, and user distributions. |

### How the backend connects the pieces

[`backend/app.py`](backend/app.py) locates and preloads WA, user-feature, conversion-score, feature-importance, and Uplift CSV files, as well as a model, from `backend/data/`. Campaign pages call [`/api/promo/overview`](backend/app.py#L208), [`/api/promo/trend`](backend/app.py#L244), and [`/api/promo/ate`](backend/app.py#L277). User pages call [`/api/user/segments`](backend/app.py#L330), [`/api/user/rfm`](backend/app.py#L384), [`/api/user/funnel`](backend/app.py#L442), and [`/api/user/profile/<id>`](backend/app.py#L661). Single-record JSON prediction uses [`/api/predict`](backend/app.py#L568); batch JSON supports up to 10,000 records via [`/api/predict/batch`](backend/app.py#L611); CSV upload classification supports up to 50,000 rows via [`/api/classify/upload`](backend/app.py#L988). Responses use a `code` / `data` JSON wrapper, with `message` on error paths.

After login, `role_required()` checks the `admin`, `analyst`, and `viewer` roles against endpoint permissions: admins can manage users; analysts can access analysis, prediction, and upload; viewers can see campaign overview and trends. Demo accounts are stored in a local JSON file. This is suitable for a local portfolio demonstration, **not** a production identity system. More request examples are in the [backend API notes (Chinese)](backend/README.md); context and methodology are in the [project notes (Chinese)](docs-project-notes.md).

### Interpretation and reproducibility limits

- WA store-sales PSM/DiD and exposure-log Uplift use **different data pipelines**. Their side-by-side display demonstrates methods; it is not three independent validations of one user cohort or experiment. Causal claims depend on data quality and identification assumptions.
- [`/api/user/segments`](backend/app.py#L330) reads offline `user_type` / `uplift_t` values from `log_uplift_result.csv`. [`/api/classify/upload`](backend/app.py#L988) instead assigns demonstration labels from conversion-score thresholds to uploaded CSV rows; it **does not re-estimate CATE** for those users. Upload labels should not be presented as causal response quadrants.
- The training script [`ub_xgboost_model.py`](scripts/04_UserBehavior分析/ub_xgboost_model.py) lists 19 features, including purchase-derived quantities, whereas the web prediction API submits six behaviour features. The public training and serving feature contracts need aligning before reliable retraining and online inference can be reproduced from raw data. The script should not be described as free of purchase-related or target-leakage risk.
- Although `smooth_prob()` receives the model's raw probability, its returned score is computed from hand-written behaviour rules rather than that probability. The online result is a **demonstration score**, not a calibrated XGBoost probability. Some API p-values and fallback estimates are also example constants; recompute and validate them before production decisions.

## Run locally

Windows `cmd.exe` example:

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
set FLASK_SECRET_KEY=replace-with-a-long-random-value
python app.py
```

In PowerShell, set `$env:FLASK_SECRET_KEY = '...'` and activate the PowerShell virtual environment; on macOS/Linux use `export FLASK_SECRET_KEY=...` and activate `.venv/bin/activate`. Choose a Python environment compatible with the [pinned dependencies](backend/requirements.txt), then open `http://localhost:5000`. A first local run creates an untracked `backend/users.json`. The demo administrator account is `demo_admin` / `demo123`; it is **only for local demonstration**. Never expose the default account or an example secret on a public server.

## Repository contents and data

- `backend/`: Flask service, plain HTML/CSS/JavaScript UI, model, and derived demonstration data needed by the application.
- `scripts/`: Offline WA Marketing Campaign, UserBehavior, PSM, DiD, Criteo, and log-Uplift work. [`criteo_conversion_model.py`](scripts/03_Criteo转化预测/criteo_conversion_model.py) is a separate conversion experiment, **not** the model currently loaded by the web service.
- `docs-project-notes.md`: Detailed project design, metrics, and methods (Chinese).

The repository excludes raw large datasets/logs, compressed backups, duplicate training outputs, and personal account databases. To retrain, obtain the corresponding datasets from their sources and point the scripts to local inputs through parameters or environment variables; do not commit downloaded raw datasets.

## License

Project code is under the [MIT License](LICENSE). Dataset and pretrained-model redistribution remains subject to each source's licence and terms.
