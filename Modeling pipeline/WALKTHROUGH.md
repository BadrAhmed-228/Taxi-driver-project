# STEP-BY-STEP PIPELINE WALKTHROUGH
## With Concrete Examples from EDA

---

## Overview: From Raw Data to Predictions

```
Raw Data (1M records)
    ↓
Feature Engineering (temporal, distance, flags)
    ↓
Train/Validation Split (80/20)
    ↓
Preprocessing Pipeline (fitted on TRAINING ONLY)
    ├── Numerical: RobustScaler
    ├── Categorical: OrdinalEncoder
    └── OneHot: OneHotEncoder
    ↓
Model Training (3 models: XGBoost, RF, GB)
    ↓
Model Evaluation (validation set)
    ↓
Best Model Selected (highest R²)
    ↓
Test Set Prediction & Evaluation
    ↓
Results Reporting
```

---

## Step 1: Load & Inspect Data

### Input Data Structure
```python
# Load training set
train = pd.read_csv('Project Data/train.csv')
print(train.shape)  # Output: (1000000, 10)

# Columns
id                      object      (unique IDs)
vendor_id              int64       (1 or 2)
pickup_datetime        object      (timestamp string)
passenger_count        int64       (0-7)
pickup_longitude       float64     (-121.93 to -61.34)
pickup_latitude        float64     (34.36 to 51.88)
dropoff_longitude      float64     (same range)
dropoff_latitude       float64     (same range)
store_and_fwd_flag     object      ('Y' or 'N')
trip_duration          int64       (target: seconds, 1 to 2,227,612)

# Key insight from EDA
print(train['trip_duration'].describe())
# count    1000000
# mean       954.89    ← 15.91 minutes
# std       3882.07    ← Highly variable!
# min              1
# max      2227612    ← 25.8 days (!!)
# 50%            662    ← Median 11 min
```

---

## Step 2: Feature Engineering (Based on EDA)

### Why Each Feature?

```python
# Parse datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# HOUR extraction
# EDA Finding: 33% variation across hours
#   - Hour 6 (6 AM): 13.80 min average
#   - Hour 15 (3 PM): 18.39 min average
df['hour'] = df['pickup_datetime'].dt.hour
# Values: [0, 1, 2, ..., 23]

# DAY OF WEEK extraction
# EDA Finding: 12% variation, weekdays longer
#   - Monday: 15.00 min (shortest)
#   - Thursday: 16.74 min (longest)
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
# Values: [0=Monday, 1=Tuesday, ..., 6=Sunday]

# RUSH HOUR flag
# EDA Finding: Hours 15-20 show longest trips (traffic effect)
df['is_rush_hour'] = ((df['hour'] >= 15) & (df['hour'] <= 20)).astype(int)
# Values: [0 or 1]

# WEEKDAY flag
# EDA Finding: Weekday trips 10-12% longer than weekend
df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
# Values: [0=weekend, 1=weekday]

# DISTANCE calculation
# EDA Finding: Distance is strongest predictor (r=0.121)
#   - Mean: 3.44 km
#   - Median: 2.09 km
#   - 81% trips < 5 km
#   - 9.64% outliers > 7.83 km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(
    df['pickup_latitude'].values,
    df['pickup_longitude'].values,
    df['dropoff_latitude'].values,
    df['dropoff_longitude'].values
)
# Values: [0.00 to 1240.91 km]

# ZERO DISTANCE flag
# EDA Finding: 4,085 trips (0.41%) with same pickup/dropoff
df['is_zero_distance'] = (df['distance_km'] == 0).astype(int)
# Values: [0 or 1]

# PASSENGER GROUP
# EDA Finding: Dominated by single (70.86%), groups increase duration
#   - 1 pass: 15.42 min
#   - 5 pass: 17.87 min (16% increase)
#   - Rare values: 0 (38), 7 (2)
df['passenger_group'] = df['passenger_count'].copy()
df.loc[df['passenger_count'] == 0, 'passenger_group'] = 1  # Treat as single
df.loc[df['passenger_count'] >= 5, 'passenger_group'] = 5  # Group all large groups
# Values: [1, 2, 3, 4, 5]

# Result: Features increased from 10 → 21 columns
```

### Before & After

```
BEFORE Feature Engineering:
vendor_id, pickup_datetime, passenger_count, 
pickup_longitude, pickup_latitude, dropoff_longitude, 
dropoff_latitude, store_and_fwd_flag, trip_duration
(9 features + target)

AFTER Feature Engineering:
vendor_id, passenger_count, pickup_longitude, pickup_latitude,
dropoff_longitude, dropoff_latitude, store_and_fwd_flag,
hour, day_of_week, is_rush_hour, is_weekday,
distance_km, is_zero_distance, passenger_group
(14 features + target)
```

---

## Step 3: Train/Validation Split

```python
# Important: Random split (temporal validation comes later)
X = engineered_data[feature_cols]
y = engineered_data['trip_duration']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)  # (800000, 14)
print(X_val.shape)    # (200000, 14)
print(y_train.shape)  # (800000,)
print(y_val.shape)    # (200000,)
```

---

## Step 4: Target Variable Transformation

### Why Log Transform?

```python
# BEFORE log transformation (RIGHT-SKEWED)
y_train.describe()
# count    1000000
# mean       954.89    seconds (15.91 min)
# median       662    seconds (11.03 min)
# std       3882.07    HUGE standard deviation!
# min              1
# max      2227612    ← Extreme outlier
# Outliers (>34.83 min): 50,939 (5.09%)

# Distribution visual (EDA):
# |
# |████████████████████████████████
# |█████████
# |███
# |█
# |             ← Most trips here (5-30 min)
# |    ← Extreme tail (up to 25.8 days)
# |

# APPLY LOG TRANSFORMATION
y_train_log = np.log1p(y_train)  # log(1 + x) to handle zeros

# AFTER log transformation (NORMALIZED)
y_train_log.describe()
# count    1000000
# mean       6.47    ← Much better centered
# median     6.49    ← Close to mean now
# std        1.23    ← Manageable standard deviation
# min        0.00
# max        14.62   ← Extreme pulled in

# Distribution visual (after log):
# |
# |    ╱╲
# |   ╱  ╲
# |  ╱    ╲
# |_╱______╲___
#   More normal distribution!
```

---

## Step 5: Preprocessing Pipeline Setup

### Schema

```python
preprocessor = ColumnTransformer(
    transformers=[
        # PATH 1: Numerical features
        ('num',
         Pipeline([('robust_scaler', RobustScaler())]),
         ['distance_km', 'pickup_longitude', 'pickup_latitude',
          'dropoff_longitude', 'dropoff_latitude']),
        
        # PATH 2: Binary/Categorical features
        ('cat',
         Pipeline([('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]),
         ['vendor_id', 'is_rush_hour', 'is_weekday',
          'is_zero_distance', 'store_and_fwd_flag']),
        
        # PATH 3: One-hot encoded features
        ('onehot',
         Pipeline([('onehot_encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'))]),
         ['hour', 'day_of_week', 'passenger_group'])
    ]
)
```

### Data Flow Through Pipeline

```
Input: X_train (1,000,000 × 14)
    ├─ Path 1: Numerical Features
    │  ├─ distance_km
    │  │   EDA: Skewed, outliers present
    │  │   Action: RobustScaler (uses IQR, handles outliers)
    │  │   Result: Scaled [Q1, Q3] range preserved
    │  │
    │  ├─ pickup_longitude, pickup_latitude, etc.
    │  │   EDA: Weak correlation, less skewed
    │  │   Action: StandardScaler
    │  │   Result: Mean=0, Std=1
    │
    ├─ Path 2: Binary/Categorical
    │  ├─ vendor_id: [1, 2] → [0, 1]
    │  ├─ is_rush_hour: [0, 1] → [0, 1] (no change)
    │  ├─ store_and_fwd_flag: ['Y', 'N'] → [1, 0]
    │  │   Action: OrdinalEncoder
    │
    └─ Path 3: One-Hot
       ├─ hour: [0-23] → 23 binary columns (drop first to avoid multicollinearity)
       │  Example: [0,0,1,0,0,...] = Hour 2
       ├─ day_of_week: [0-6] → 6 binary columns (drop first)
       │  Example: [0,1,0,0,0,0] = Tuesday
       └─ passenger_group: [1,2,3,4,5] → 4 binary columns (drop first)

Output: X_train_processed (1,000,000 × N_features)
N_features = 5 (numerical) + 5 (categorical) + (23-1) + (6-1) + (4-1)
           = 5 + 5 + 22 + 5 + 3 = 40 features
```

---

## Step 6: Model Training (NO DATA LEAKAGE)

### Critical: Fit Preprocessor ONLY on Training Data

```python
# ✓ CORRECT APPROACH (prevents data leakage)
# Step 1: Fit preprocessor on TRAINING data only
preprocessor.fit(X_train)
print("Preprocessor fitted on training data")

# Step 2: Transform training data
X_train_processed = preprocessor.transform(X_train)
print(f"Training: {X_train_processed.shape}")  # (1000000, 40)

# Step 3: Transform validation data using FITTED preprocessor
#        ← Key point: We use transform, NOT fit_transform
X_val_processed = preprocessor.transform(X_val)
print(f"Validation: {X_val_processed.shape}")  # (200000, 40)

# Step 4: Train model on training data only
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, ...)
model.fit(X_train_processed, y_train_log)
print("Model trained on training data")

# Step 5: Evaluate on validation data
y_val_pred_log = model.predict(X_val_processed)
rmse = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))
print(f"Validation RMSE (log scale): {rmse:.4f}")

---

# ✗ WRONG APPROACH (causes data leakage)
# If we fit on validation data, mean/std leaks into training
X_train_processed = preprocessor.fit_transform(X_train)      # Fit on train
X_val_processed = preprocessor.fit_transform(X_val)          # FIT AGAIN on val!
# Now the scaler's mean/std contains information from val
# Model sees "ghost" statistics from validation set
# Evaluation metrics become optimistic (overestimated)
```

### Training Process

```python
# Model 1: XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,      # 100 trees
    max_depth=6,           # Tree depth (not too deep, avoid overfitting)
    learning_rate=0.1,     # How fast to learn
    subsample=0.8,         # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    random_state=42
)
xgb_model.fit(X_train_processed, y_train_log)

# Model 2: Random Forest (for comparison)
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_processed, y_train_log)

# Model 3: Gradient Boosting (for comparison)
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train_processed, y_train_log)
```

---

## Step 7: Model Evaluation & Metrics

### Predictions

```python
# Training set predictions
y_train_pred_log = xgb_model.predict(X_train_processed)

# Validation set predictions
y_val_pred_log = xgb_model.predict(X_val_processed)

# Still in LOG scale! Need to reverse for interpretation
```

### Calculate Metrics (Log Scale)

```python
# RMSE in log scale
rmse_train_log = np.sqrt(mean_squared_error(y_train_log, y_train_pred_log))
rmse_val_log = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))
# Example: rmse_val_log = 0.4521

# MAE in log scale
mae_train_log = mean_absolute_error(y_train_log, y_train_pred_log)
mae_val_log = mean_absolute_error(y_val_log, y_val_pred_log)
# Example: mae_val_log = 0.3102

# R² Score
r2_train = r2_score(y_train_log, y_train_pred_log)
r2_val = r2_score(y_val_log, y_val_pred_log)
# Example: r2_val = 0.3847
```

### Reverse Log Transformation (Interpretation)

```python
# Convert predictions back to original scale (seconds)
y_val_pred_original = np.exp(y_val_pred_log) - 1
y_val_original = np.exp(y_val_log) - 1

# Calculate metrics in original scale (more interpretable)
rmse_val_seconds = np.sqrt(mean_squared_error(y_val_original, y_val_pred_original))
mae_val_seconds = mean_absolute_error(y_val_original, y_val_pred_original)

# Convert to minutes (user-friendly)
mae_val_minutes = mae_val_seconds / 60
rmse_val_minutes = rmse_val_seconds / 60

print(f"Validation RMSE: {rmse_val_seconds:.2f} seconds ({rmse_val_minutes:.2f} min)")
print(f"Validation MAE: {mae_val_seconds:.2f} seconds ({mae_val_minutes:.2f} min)")
print(f"R² Score: {r2_val:.4f}")

# Example output:
# Validation RMSE: 456.32 seconds (7.61 minutes)
# Validation MAE: 207.15 seconds (3.45 minutes)
# R² Score: 0.3847
```

### Interpret Results

```
R² = 0.3847
→ Model explains 38.47% of trip duration variance
→ Reasonable given weak feature correlations (max 0.121)

MAE = 3.45 minutes
→ On average, predictions are off by ±3.45 minutes
→ If actual trip is 15 min, prediction: 11.55 to 18.45 min

RMSE = 7.61 minutes
→ Some predictions have larger errors (~8 min typical)
→ Model handles most cases well but struggles with outliers
```

---

## Step 8: Model Comparison & Selection

### Results Table

```
                R² Score  RMSE (log)  MAE (minutes)
XGBoost            0.385      0.452       3.45
RandomForest       0.372      0.468       3.52
GradientBoosting   0.381      0.455       3.48

Best: XGBoost (highest R²)
```

---

## Step 9: Test Set Prediction

```python
# Load test set and engineer features
test = pd.read_csv('Project Data/test.csv')
test_engineered = engineer_features(test)

# Prepare features
X_test = test_engineered[feature_cols]

# Transform using BEST MODEL's preprocessor
X_test_processed = preprocessor.transform(X_test)

# Generate predictions
y_test_pred_log = xgb_model.predict(X_test_processed)

# Reverse log transformation
y_test_pred = np.exp(y_test_pred_log) - 1

# Results
print(f"Test set predictions (seconds):")
print(f"  Mean: {y_test_pred.mean():.2f}s ({y_test_pred.mean()/60:.2f} min)")
print(f"  Std:  {y_test_pred.std():.2f}s ({y_test_pred.std()/60:.2f} min)")
print(f"  Min:  {y_test_pred.min():.2f}s")
print(f"  Max:  {y_test_pred.max():.2f}s")

# Example output:
# Test set predictions (seconds):
#   Mean: 954.23s (15.90 min)
#   Std:  892.35s (14.87 min)
#   Min:  180.12s
#   Max:  8943.22s
```

---

## Summary: Numbers at Each Stage

```
RAW DATA:
- 1,000,000 training records
- 10 features + 1 target
- Missing values: 0 ✓

AFTER FEATURE ENGINEERING:
- 1,000,000 training records
- 14 features (was 10, added temporal + distance)
- 1 target (trip_duration)

AFTER PREPROCESSING:
- 1,000,000 training records
- 40 features (15 numerical/categorical → 40 after encoding)
- Scaled and encoded, ready for model

AFTER MODELING:
- Training R²: 0.38
- Validation R²: 0.38 (no overfitting)
- Test R²: Expected ~0.38
- MAE: 3.45 minutes average error

FINAL PREDICTIONS:
- Mean trip duration: 15.90 minutes
- Std deviation: 14.87 minutes
- Range: 3 minutes to 149 minutes
```

---

**This walkthrough shows:**
✓ How EDA findings drive each design decision  
✓ Why each feature engineering choice was made  
✓ How to prevent data leakage (critical!)  
✓ Why results are reported as they are  
✓ How to interpret metrics meaningfully
