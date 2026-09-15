# Kuaishou User Activity Prediction

> Predicting whether a user will remain active on the Kuaishou short-video app in a future time window, using raw behavioral logs. Built for the **2018 China Collegiate Computing Contest – Big Data Challenge (大数据挑战赛)**.

---

## 📌 Overview

| Item | Detail |
|------|--------|
| **Competition** | 2018 China Collegiate Computing Contest — Big Data Challenge |
| **Host** | Kuaishou (快手) |
| **Task Type** | Binary classification on imbalanced, large-scale behavioral logs |
| **Target** | Will a registered user launch the app in the prediction window? |
| **Metric** | AUC (Area Under the ROC Curve) |
| **Core Technique** | Massive hand-crafted feature engineering + XGBoost |
| **Language** | Python (pandas / numpy / xgboost / lightgbm / scikit-learn) |

---

## 🧠 Problem Statement

Short-video platforms lose revenue when users churn. Given anonymized, time-stamped logs of app launches, video views, video uploads, and registrations, the goal is to **predict each user's future activity** (binary label: active / inactive in the next window).

The challenge is threefold:
1. **No structured features** — everything must be aggregated from raw event logs.
2. **Temporal leakage** — features must only use past behavior; validation must respect time order.
3. **Scale** — millions of interaction records require memory- and compute-aware engineering.

---

## 📂 Dataset

Four tab-separated log files (no headers, column names assigned in code):

| File | Schema | Description |
|------|--------|-------------|
| `user_register_log.txt` | `user_id, reg_day, reg_type, device_type` | Registration metadata |
| `app_launch_log.txt` | `user_id, day` | App open events |
| `user_activity_log.txt` | `user_id, day, page, video_id, author_id, action_type` | Watch / like / comment / etc. |
| `video_create_log.txt` | `user_id, day` | User-uploaded video events |

---

## 🏗️ Solution Architecture

```
            ┌─────────────────────────────────────────────┐
            │           Raw Log Files (4 tables)          │
            └──────────────────────┬──────────────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │  Time-Window Split (sliding window) │
                │  train: day 16–23  | test: day ≥ 24 │
                └──────────────────┬──────────────────┘
                                   │
   ┌───────────────┬───────────────┼───────────────┬───────────────┐
   ▼               ▼               ▼               ▼               ▼
[Register]    [App Launch]    [Activity]      [Video Create]   [Cross]
   │               │               │               │               │
   └─── Feature Engineering (per-user aggregations) ───────────────┘
                                   │
                          ┌────────▼────────┐
                          │  ~300+ features  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   XGBoost GBDT  │ ← AUC evaluation
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  Probabilities  │ → submission file
                          └─────────────────┘
```

---

## 🔧 Feature Engineering (Core Contribution)

The bulk of this project is **hand-crafted feature engineering**. For each user (and author, when the user is also a creator), features are aggregated across multiple statistical dimensions. A representative subset:

### 1. App Launch Features (`App`)
- Launch counts, unique active days, launch frequency (count / window size)
- Max / min / mean / median / var / skew / kurtosis of active days
- Last-launch-day rank and distance to window end
- Recent-window counts (1-day, 2-day, 3-day lag features)
- Time-difference distribution between each launch and the prediction day

### 2. User Activity Features (`Act`)
- Video watch counts, unique videos, unique authors, repeat-watch indicators
- Per-action-type counts and ratios (6 action types: 0–5)
- Per-page counts and ratios (5 pages: 0–4)
- Date distribution statistics (max / min / mean / median / var / skew / kurt)
- **Author-side features** (when the user is a creator): times watched, unique watchers, action-type & page breakdowns
- **Heat features**: popularity statistics of the videos / authors a user interacted with
- Full 3-day-recent behavior mirror of all the above (short-term trend signals)

### 3. Video Creation Features (`Video`)
- Number of videos created, creation-date statistics
- Last-day and 3-day recent creation counts

### 4. Registration Features (`Reg`)
- One-hot encoding of registration channel (`reg_type`)
- Device-type coarse-grained bucketing into 4 tiers (based on frequency tiers)
- Device-type registration counts
- Time gaps: registration → first activity, registration → mean activity day, registration → window end

### Feature Design Principles
- **Time-aware**: every feature is computed only from the user's behavior *before* the prediction window — no leakage.
- **Multi-granularity**: global window stats + short-term (1/2/3-day) recent stats to capture both habit and trend.
- **Distributional**: not just counts — skew / kurtosis / variance capture behavioral irregularity.
- **Cross-entity**: a user's watch behavior is enriched by *the popularity of what they watched* (heat features).

---

## 🤖 Modeling

**Algorithm**: XGBoost (gradient-boosted decision trees), `binary:logistic` objective.

| Hyperparameter | Value | Rationale |
|----------------|-------|----------|
| `eta` | 0.02 | Small learning rate for stable boosting |
| `max_depth` | 5 | Shallow trees to prevent overfitting |
| `subsample` | 0.8 | Row sampling for regularization |
| `colsample_bytree` | 0.8 | Column sampling for regularization |
| `min_child_weight` | 18 | Conservative leaf growth on imbalanced data |
| `num_boost_round` | 600–700 | Sufficient ensemble size for low `eta` |

Also imports **LightGBM** and scikit-learn utilities (`SelectFromModel`, `RFE`, `MinMaxScaler`, `GradientBoostingClassifier`) for experimentation and feature selection.

---

## ⏱️ Validation Strategy

A **time-based sliding window** is used instead of random K-fold — the correct choice for temporal data:

| Split | Feature Window | Label Window |
|-------|----------------|--------------|
| **Train** | day 16 – 23 | day ≥ 24 (registered by day 23) |
| **Validation** | day 15 – 22 | day 21 – 27 (registered by day 20) |
| **Test** | day ≥ 24 | future window |

This simulates production: train on the past, predict the future, and ensures no temporal leakage between features and labels.

---

## ⚙️ Engineering Highlights

- **Multiprocessing** — `multiprocessing.Pool` parallelizes train/test feature pipelines (`ForTrain` / `ForTest` run concurrently).
- **Memory-aware aggregation** — heavy use of `pd.pivot_table` and in-place column reuse; `gc` module invoked to manage RAM on large logs.
- **Vectorized statistics** — custom `getkurt` / `getvar` / `getskew` helpers passed as pivot aggfuncs to compute distributional features in a single pass.
- **Modular pipeline** — each data source (`App`, `Act`, `Video`, `Reg`) is a self-contained function that merges into a shared user table, making the pipeline easy to extend.

---

## 📁 Project Structure

```
MarketData/
└── github/
    ├── main.py                 # End-to-end pipeline: features → model → submission
    ├── README.md               # This file
    └── (external data files — not committed)
        ├── user_register_log.txt
        ├── app_launch_log.txt
        ├── user_activity_log.txt
        └── video_create_log.txt
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy xgboost lightgbm scikit-learn matplotlib seaborn

# 2. Place the 4 raw .txt log files in the parent directory of main.py
#    (the code reads them via r'..\xxx_log.txt')

# 3. Run validation pipeline (offline AUC evaluation)
python -c "from main import Mul_val; Mul_val()"

# 4. Run test pipeline (generates submission)
python -c "from main import Mul_test; Mul_test()"
```

Outputs:
- `..\train724.csv`, `..\val724.csv` — engineered feature tables
- `FatKong724.txt` — submission file (`user_id, probability`)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![pandas](https://img.shields.io/badge/pandas-dataframe-150458)
![XGBoost](https://img.shields.io/badge/XGBoost-GBDT-ff9900)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-00b388)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![multiprocessing](https://img.shields.io/badge/multiprocessing-parallel-3776AB)

---

## 📝 Key Takeaways

- **Feature engineering dominates this task** — tree-boosting models benefit far more from behavior-distribution features than from hyperparameter tuning.
- **Time-aware design matters** — a naive random split would inflate AUC through temporal leakage; the sliding-window setup reflects a real deployment scenario.
- **Statistical depth pays off** — beyond simple counts, skew / kurtosis / variance of action timestamps proved to be strong behavioral signals.

---

## 📜 License

Personal project for educational and portfolio purposes. Dataset © Kuaishou / competition organizers.
