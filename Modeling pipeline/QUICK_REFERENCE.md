# QUICK REFERENCE GUIDE
## Taxi Trip Duration Pipeline

---

## 🚀 Running the Pipeline

```bash
cd "c:\Users\pc\Downloads\Taxi driver project"
python modeling_pipeline.py
```

---

## 📊 Key Design Decisions (EDA-Driven)

| Decision | EDA Finding | Implementation |
|----------|------------|-----------------|
| **Log Transform Target** | Right-skewed, 5.09% outliers, max 25.8 days | `np.log1p(trip_duration)` |
| **RobustScaler Distance** | Skewed (std 4.42 km), 9.64% outliers | `RobustScaler()` (not StandardScaler) |
| **OneHot Hour/Day** | 33% hourly variation, 12% daily variation | `OneHotEncoder(drop='first')` |
| **Keep Outliers** | 50,939 duration outliers are valid (stored rides) | Don't remove; let RobustScaler handle |
| **Tree Models** | Weak correlations (0.121 max), non-linear | XGBoost, RandomForest, GradientBoosting |
| **Feature Engineering** | Multiple patterns (temporal, distance, vendor) | Extract hour, day_of_week, distance_km, flags |
| **Data Leakage Prevention** | Critical for honest evaluation | Fit preprocessor ONLY on training data |

---

## ⚙️ Feature Pipeline Breakdown

### Input Features
```
vendor_id, pickup_datetime, passenger_count, 
pickup_longitude, pickup_latitude,
dropoff_longitude, dropoff_latitude,
store_and_fwd_flag, trip_duration (target)
```

### Engineered Features
```
✓ hour (0-23)                    → OneHot
✓ day_of_week (0-6)             → OneHot
✓ is_rush_hour (0/1)            → Ordinal
✓ is_weekday (0/1)              → Ordinal
✓ distance_km (calculated)      → RobustScaler
✓ is_zero_distance (0/1)        → Ordinal
✓ passenger_group (grouped)     → OneHot
✓ vendor_id (1/2)               → Ordinal
✓ store_and_fwd_flag (Y/N)      → Ordinal
```

### Numerical → RobustScaler
```python
[
    'distance_km',
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude'
]
```

### Categorical → OrdinalEncoder
```python
[
    'vendor_id',
    'store_and_fwd_flag',
    'is_rush_hour', 'is_weekday', 'is_zero_distance'
]
```

### OneHot → OneHotEncoder (drop='first')
```python
[
    'hour', 'day_of_week', 'passenger_group'
]
```

---

## 📈 Expected Outputs

### Training Console Output
```
==================================================================
FEATURE ENGINEERING
==================================================================
Engineered features added. Train shape: (1000000, 21)

==================================================================
CREATING PREPROCESSING PIPELINE
==================================================================
Numerical features (5): [...]
Categorical/Binary features (5): [...]
One-hot features (3): [...]

==================================================================
Training XGBoost
==================================================================
Processed training shape: (1000000, N_features)
Processed validation shape: (229319, N_features)
Training XGBoost...

Validation Set Metrics (Log Scale):
  RMSE (log): 0.XX
  MAE (log):  0.XX
  R² Score:   0.XXX

Validation Set Metrics (Original Scale - Seconds):
  RMSE: XXXX.XX seconds (XX.XX minutes)
  MAE:  XXXX.XX seconds (XX.XX minutes)

==================================================================
MODEL COMPARISON - VALIDATION SET
==================================================================
             R² Score  RMSE (log)  MAE (minutes)
XGBoost         0.XX      0.XX        XX.XX
RandomForest    0.XX      0.XX        XX.XX
GradientBoosting 0.XX     0.XX        XX.XX

🏆 Best Model: XGBoost (R² = 0.XXX)

==================================================================
TEST SET EVALUATION (Final Model)
==================================================================
Test predictions summary:
  Mean predicted duration: XXXX.XX seconds (XX.XX minutes)
  Std predicted duration: XXXX.XX seconds
  Min: XXXX.XXs, Max: XXXX.XXs

==================================================================
PIPELINE COMPLETE ✓
==================================================================
```

---

## 🔍 Understanding Results

### Metrics Interpretation

**R² Score (primary metric)**
- 0.0 = Model is no better than predicting mean
- 0.3 - 0.5 = Good (typical for this dataset)
- 1.0 = Perfect prediction (unrealistic)

**MAE in Minutes (interpretable)**
- If MAE = 3 minutes: On average, prediction is off by 3 minutes
- User-friendly: "We predict trip duration within ±3 minutes"

**RMSE**
- Larger errors penalized more heavily
- Useful for detecting outlier prediction errors

### Example Interpretation
```
Validation Set Results:
  R² = 0.35
  MAE = 3.45 minutes
  RMSE = 7.82 minutes

Translation:
- Model explains 35% of trip duration variance
- Average prediction error: ±3.45 minutes
- Some trips predicted with ~8 minute error
- Weak features (max correlation 0.121) limit performance
```

---

## 🛡️ Data Leakage Prevention Checklist

✓ Preprocessor fitted ONLY on training data  
✓ Validation/test use `.transform()` not `.fit_transform()`  
✓ Scaler statistics (mean, std, IQR) computed from training only  
✓ Model trained on training data only  
✓ Validation evaluation uses fitted preprocessor  
✓ No information from validation set influences model  

**Code Pattern:**
```python
# ✓ CORRECT
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)  # No fit!

# ✗ WRONG
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.fit_transform(X_val)  # Double fit!
```

---

## 🔧 Customization Guide

### Change Target Transformation
```python
# Current: Log transformation
y_train_log = np.log1p(y_train)

# Alternative: Box-Cox (if y > 0)
from scipy.stats import boxcox
y_train_transformed, lambda_param = boxcox(y_train)

# Alternative: No transformation (not recommended for this data)
y_train_log = y_train
```

### Change Scaler for Distance
```python
# Current: RobustScaler
('robust_scaler', RobustScaler())

# Alternative: QuantileTransformer (uniform distribution)
('quantile', QuantileTransformer(output_distribution='uniform'))

# Alternative: PowerTransformer (similar to Box-Cox)
('power', PowerTransformer(method='yeo-johnson'))
```

### Adjust XGBoost Hyperparameters
```python
# Current (balanced)
xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
)

# Deeper model (more complex, longer training)
xgb.XGBRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
)

# Simpler model (faster, less overfit risk)
xgb.XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.15,
)
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'xgboost'`
```bash
pip install xgboost
# or
conda install -c conda-forge xgboost
```

### Issue: Data shape mismatch in prediction
**Problem:** Test set has different features
```python
# Ensure test features match training
test_features = test[['vendor_id', 'pickup_datetime', ...]]
# Must include ALL engineered features before preprocessing
```

### Issue: Predictions seem unrealistic (too high/low)
**Check:** Are you reversing the log transformation?
```python
# ✓ CORRECT
y_pred = np.exp(y_pred_log) - 1

# ✗ WRONG (forgot to reverse)
y_pred = y_pred_log  # These are log values, not seconds!
```

---

## 📚 File Dependencies

```
c:\Users\pc\Downloads\Taxi driver project\
├── modeling_pipeline.py          ← Main script
├── PIPELINE_DESIGN.md            ← Design rationale
├── QUICK_REFERENCE.md            ← This file
├── EDA_REPORT.md                 ← Source of all decisions
└── Project Data\
    ├── train.csv                 ← 1M records
    ├── val.csv                   ← 229K records
    └── test.csv                  ← 229K records
```

---

## 🎯 Success Criteria

✓ Zero data leakage (train-only fitting)  
✓ All preprocessing decisions based on EDA  
✓ Models trained and compared  
✓ Metrics reported (R², RMSE, MAE)  
✓ Predictions reversed from log scale  
✓ Test set evaluated with best model  

**Expected Performance:**
- R² = 0.3 - 0.5 (weak features limit performance)
- MAE ≈ 3-4 minutes (±3-4 min average error)
- Consistent across train/val sets (no overfitting)

---

**Version:** 1.0  
**Created:** 2026-04-18  
**Based on:** EDA Report Findings
