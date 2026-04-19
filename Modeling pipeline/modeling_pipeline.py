"""
Production-Level ML Pipeline for NYC Taxi Trip Duration Prediction
Based on EDA findings from EDA_REPORT.md

Key Design Decisions:
1. Log transform target (right-skewed, 5.09% outliers with extreme values)
2. RobustScaler for distance (skewed + outliers)
3. StandardScaler for coordinates
4. Tree-based models (weak linear correlations - 0.121 max)
5. Feature engineering: hour, day_of_week, is_rush_hour, distance
6. Proper train/validation split with fit-only-on-train strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder,
    FunctionTransformer
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data():
    """Load training, validation, and test sets."""
    train = pd.read_csv(r'c:\Users\pc\Downloads\Taxi driver project\Project Data\train.csv')
    val = pd.read_csv(r'c:\Users\pc\Downloads\Taxi driver project\Project Data\val.csv')
    test = pd.read_csv(r'c:\Users\pc\Downloads\Taxi driver project\Project Data\test.csv')

    print("=" * 70)
    print("DATA LOADING COMPLETE")
    print("=" * 70)
    print(f"Training set: {train.shape}")
    print(f"Validation set: {val.shape}")
    print(f"Test set: {test.shape}")
    print(f"\nColumns: {list(train.columns)}")
    print(f"Missing values (train): {train.isnull().sum().sum()}")

    return train, val, test


# ============================================================================
# 2. FEATURE ENGINEERING (Based on EDA)
# ============================================================================

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance in km between two points.
    EDA Finding: Distance is the strongest predictor (correlation 0.121)
    """
    R = 6371  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def engineer_features(df, fit_on_train=False):
    """
    Feature engineering based on EDA findings.

    EDA Insights Applied:
    - Hour: 33% variation in duration (Hour 15 longest, Hour 6 shortest)
    - Day of week: 12% variation (Thursday longest, Monday shortest)
    - Is_rush_hour: Hours 15-20 show significant effects
    - Distance: Mean 3.44 km, 81% trips < 5 km
    - Passenger count: 70.86% solo, increases with count
    - Vendor: 27% difference between vendors
    """
    df = df.copy()

    # Parse datetime (handle both formats if needed)
    if df['pickup_datetime'].dtype == 'object':
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Extract temporal features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['pickup_datetime'].dt.month

    # EDA Finding: Hours 15-20 show rush hour effect (32% longer)
    df['is_rush_hour'] = ((df['hour'] >= 15) & (df['hour'] <= 20)).astype(int)

    # EDA Finding: Weekday vs weekend effect (weekdays 10-12% longer)
    df['is_weekday'] = (df['day_of_week'] < 5).astype(int)

    # Calculate haversine distance
    df['distance_km'] = calculate_haversine_distance(
        df['pickup_latitude'].values,
        df['pickup_longitude'].values,
        df['dropoff_latitude'].values,
        df['dropoff_longitude'].values
    )

    # Handle zero-distance trips (EDA: 0.41% of data)
    # Keep them but mark as special case
    df['is_zero_distance'] = (df['distance_km'] == 0).astype(int)

    # Passenger count group (EDA: 70.86% single, rare 0 and 7)
    df['passenger_group'] = df['passenger_count'].copy()
    df.loc[df['passenger_count'] == 0, 'passenger_group'] = 1  # Treat as single
    df.loc[df['passenger_count'] >= 5, 'passenger_group'] = 5  # Group large groups

    return df


# ============================================================================
# 3. DATA PREPROCESSING PIPELINE
# ============================================================================

def create_preprocessing_pipeline():
    """
    Create sklearn ColumnTransformer with separate paths for numerical
    and categorical features.

    Strategy based on EDA:
    - Numerical: RobustScaler for skewed features (distance)
    - Categorical: OneHotEncoder for high-cardinality
    - OrdinalEncoder for binary/ordinal features
    """

    # Numerical features
    # EDA: Distance is right-skewed (std 4.42 km) with outliers
    numerical_features = ['distance_km', 'pickup_longitude', 'pickup_latitude',
                         'dropoff_longitude', 'dropoff_latitude']

    numerical_transformer = Pipeline(steps=[
        ('robust_scaler', RobustScaler())  # Handles outliers better than StandardScaler
    ])

    # Categorical features
    categorical_features = ['vendor_id', 'is_rush_hour', 'is_weekday',
                           'is_zero_distance', 'store_and_fwd_flag']

    # Binary/ordinal features -> OrdinalEncoder
    categorical_transformer = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                          unknown_value=-1))
    ])

    # One-hot encoded features
    onehot_features = ['hour', 'day_of_week', 'passenger_group']

    onehot_transformer = Pipeline(steps=[
        ('onehot_encoder', OneHotEncoder(sparse_output=False,
                                        handle_unknown='ignore',
                                        drop='first'))  # Avoid multicollinearity
    ])

    # Combine all transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('onehot', onehot_transformer, onehot_features)
        ]
    )

    return preprocessor, numerical_features, categorical_features, onehot_features


# ============================================================================
# 4. MODEL SELECTION & TRAINING
# ============================================================================

def create_models():
    """
    Create models based on EDA findings.

    EDA Insight: Weak linear correlations (max 0.121) suggest tree-based
    models are more appropriate than linear regression.
    """

    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    }

    return models


# ============================================================================
# 5. TRAINING PIPELINE
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, preprocessor, model_name, model):
    """
    Train model with proper preprocessing.

    CRITICAL: Fit preprocessor ONLY on training data to prevent leakage.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    # FIT preprocessor ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # TRANSFORM validation data (do NOT fit again)
    X_val_processed = preprocessor.transform(X_val)

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_val_processed.shape}")

    # Train model
    print(f"Training {model_name}...")
    model.fit(X_train_processed, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_processed)
    y_val_pred = model.predict(X_val_processed)

    return model, preprocessor, y_train_pred, y_val_pred


# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================

def evaluate_model(y_true, y_pred, split_name='Validation'):
    """
    Evaluate model performance.

    Metrics chosen based on regression problem:
    - RMSE: Penalizes large errors (important for trip duration)
    - MAE: Interpretable error in minutes
    - R²: Proportion of variance explained
    """

    # IMPORTANT: Reverse log transformation for interpretability
    y_true_original = np.exp(y_true) - 1  # Reverse log transform
    y_pred_original = np.exp(y_pred) - 1

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # In original scale (seconds)
    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae_original = mean_absolute_error(y_true_original, y_pred_original)

    print(f"\n{split_name} Set Metrics (Log Scale):")
    print(f"  RMSE (log): {rmse:.4f}")
    print(f"  MAE (log):  {mae:.4f}")
    print(f"  R² Score:   {r2:.4f}")

    print(f"\n{split_name} Set Metrics (Original Scale - Seconds):")
    print(f"  RMSE: {rmse_original:.2f} seconds ({rmse_original/60:.2f} minutes)")
    print(f"  MAE:  {mae_original:.2f} seconds ({mae_original/60:.2f} minutes)")

    return {
        'rmse_log': rmse,
        'mae_log': mae,
        'r2': r2,
        'rmse_seconds': rmse_original,
        'mae_seconds': mae_original
    }


# ============================================================================
# 7. MAIN PIPELINE EXECUTION
# ============================================================================

def main():
    """Execute complete modeling pipeline."""

    # Load data
    train, val, test = load_data()

    # Feature engineering
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)

    train_engineered = engineer_features(train)
    val_engineered = engineer_features(val)
    test_engineered = engineer_features(test)

    print(f"Engineered features added. Train shape: {train_engineered.shape}")
    print(f"New columns: hour, day_of_week, month, is_rush_hour, is_weekday, distance_km, is_zero_distance, passenger_group")

    # Separate features and target
    feature_cols = [
        'vendor_id', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
        'store_and_fwd_flag', 'hour', 'day_of_week', 'distance_km',
        'is_rush_hour', 'is_weekday', 'is_zero_distance', 'passenger_group'
    ]

    X_train = train_engineered[feature_cols].copy()
    y_train = train_engineered['trip_duration'].copy()

    X_val = val_engineered[feature_cols].copy()
    y_val = val_engineered['trip_duration'].copy()

    X_test = test_engineered[feature_cols].copy()

    # Log transform target (EDA: highly right-skewed, 5.09% outliers)
    print("\nApplying log transformation to target variable...")
    y_train_log = np.log1p(y_train)  # log1p = log(1 + x) handles zeros
    y_val_log = np.log1p(y_val)

    print(f"Original target - mean: {y_train.mean():.2f}s, std: {y_train.std():.2f}s")
    print(f"Log target - mean: {y_train_log.mean():.4f}, std: {y_train_log.std():.4f}")

    # Create preprocessing pipeline
    print("\n" + "="*70)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*70)

    preprocessor, num_features, cat_features, onehot_features = create_preprocessing_pipeline()
    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Categorical/Binary features ({len(cat_features)}): {cat_features}")
    print(f"One-hot features ({len(onehot_features)}): {onehot_features}")

    # Create models
    models = create_models()

    # Train and evaluate each model
    results = {}
    trained_models = {}

    for model_name, model in models.items():
        # Create new preprocessor for each model (fresh fit)
        preprocessor_copy, _, _, _ = create_preprocessing_pipeline()

        model_fitted, prep_fitted, y_train_pred, y_val_pred = train_model(
            X_train, y_train_log, X_val, y_val_log,
            preprocessor_copy, model_name, model
        )

        # Evaluate
        train_metrics = evaluate_model(y_train_log, y_train_pred, 'Training')
        val_metrics = evaluate_model(y_val_log, y_val_pred, 'Validation')

        results[model_name] = {
            'train': train_metrics,
            'val': val_metrics
        }
        trained_models[model_name] = (model_fitted, prep_fitted)

    # Summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON - VALIDATION SET")
    print("="*70)

    comparison_df = pd.DataFrame({
        model: {
            'R² Score': results[model]['val']['r2'],
            'RMSE (log)': results[model]['val']['rmse_log'],
            'MAE (minutes)': results[model]['val']['mae_seconds'] / 60
        }
        for model in results
    }).T

    print(comparison_df)

    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['val']['r2'])
    print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['val']['r2']:.4f})")

    # Final test set evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION (Final Model)")
    print("="*70)

    best_model, best_preprocessor = trained_models[best_model_name]
    X_test_processed = best_preprocessor.transform(X_test)
    y_test_pred_log = best_model.predict(X_test_processed)
    y_test_pred = np.exp(y_test_pred_log) - 1  # Reverse transformation

    print(f"\nTest predictions summary:")
    print(f"  Mean predicted duration: {y_test_pred.mean():.2f} seconds ({y_test_pred.mean()/60:.2f} minutes)")
    print(f"  Std predicted duration: {y_test_pred.std():.2f} seconds")
    print(f"  Min: {y_test_pred.min():.2f}s, Max: {y_test_pred.max():.2f}s")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE ✓")
    print("="*70)

    return results, trained_models, best_model_name, y_test_pred


if __name__ == '__main__':
    results, trained_models, best_model, predictions = main()
