"""
Production-Level ML Pipeline for NYC Taxi Trip Duration Prediction
Based on EDA findings from EDA_REPORT.md

Key Design Decisions:
1. Log transform target (right-skewed, 5.09% outliers with extreme values)
2. RobustScaler for distance (skewed + outliers)
3. StandardScaler for coordinates
4. Ridge regression with alpha=1
5. Feature engineering: hour, day_of_week, is_rush_hour, distance
6. Proper train/validation split with fit-only-on-train strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import joblib
import os

from sklearn.preprocessing import (
    StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder,
    FunctionTransformer, PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    # Feature interactions (Idea 2)
    df['distance_hour_interaction'] = df['distance_km'] * df['hour']
    df['distance_rush_interaction'] = df['distance_km'] * df['is_rush_hour']
    df['lat_lon_distance'] = df['pickup_latitude'] * df['pickup_longitude']

    # Cyclic time features (Idea 3)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Distance-based features (Idea 3)
    df['distance_log'] = np.log1p(df['distance_km'])
    df['distance_category'] = pd.cut(df['distance_km'],
        bins=[0, 1, 3, 5, 10, 20, 50, float('inf')],
        labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extreme', 'outlier'])

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
    - Polynomial: Degree 2 for selected features to capture non-linear relationships
    """

    # Numerical features - Advanced preprocessing (Idea 2)
    numerical_features = ['distance_km', 'pickup_longitude', 'pickup_latitude',
                         'dropoff_longitude', 'dropoff_latitude', 'distance_log']

    numerical_transformer = Pipeline(steps=[
        ('power', PowerTransformer(method='yeo-johnson')),  # Better for skewed data
        ('scaler', StandardScaler())
    ])

    # Polynomial features (Idea 1) - degree 2 for key continuous features
    polynomial_features = ['distance_km', 'pickup_latitude', 'pickup_longitude',
                          'distance_hour_interaction', 'distance_rush_interaction']

    polynomial_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
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
            ('poly', polynomial_transformer, polynomial_features),
            ('cat', categorical_transformer, categorical_features),
            ('onehot', onehot_transformer, onehot_features)
        ]
    )

    return preprocessor, numerical_features, polynomial_features, categorical_features, onehot_features


# ============================================================================
# 4. MODEL SELECTION & TRAINING
# ============================================================================

def create_models():
    """
    Create Ridge model with alpha=1 based on requirements.
    """

    models = {
        'Ridge': Ridge(
            alpha=1.0,
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

def evaluate_model(y_true, y_pred, split_name='Validation', y_train_log=None):
    """
    Evaluate model performance.

    Metrics chosen based on regression problem:
    - RMSE: Penalizes large errors (important for trip duration)
    - MAE: Interpretable error in minutes
    - R²: Proportion of variance explained

    Parameters:
    -----------
    y_true : array-like
        True target values (in log space)
    y_pred : array-like
        Predicted values (in log space)
    split_name : str
        Name of the split (e.g., 'Validation', 'Training')
    y_train_log : array-like, optional
        Training log values for calculating bounds for clipping
    """

    # Clip predictions to reasonable bounds to prevent numerical overflow
    if y_train_log is not None:
        log_mean = y_train_log.mean()
        log_std = y_train_log.std()
        min_bound = log_mean - 3 * log_std
        max_bound = log_mean + 3 * log_std
    else:
        log_mean = y_true.mean()
        log_std = y_true.std()
        min_bound = log_mean - 3 * log_std
        max_bound = log_mean + 3 * log_std

    y_pred_clipped = np.clip(y_pred, min_bound, max_bound)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred_clipped)
    r2 = r2_score(y_true, y_pred_clipped)

    # IMPORTANT: Reverse log transformation for interpretability
    # Use clipped predictions to prevent numerical overflow
    y_true_original = np.expm1(y_true)  # Stable reverse of log1p
    y_pred_original = np.expm1(y_pred_clipped)

    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae_original = mean_absolute_error(y_true_original, y_pred_original)

    print(f"\n{split_name} Set Metrics:")
    print(f"  R² Score:   {r2:.4f}")
    print(f"  RMSE (log): {rmse:.4f}")
    print(f"  MAE (log):  {mae:.4f}")
    print(f"  RMSE (seconds): {rmse_original:.2f} ({rmse_original/60:.2f} minutes)")
    print(f"  MAE (seconds):  {mae_original:.2f} ({mae_original/60:.2f} minutes)")

    return {
        'rmse_log': rmse,
        'mae_log': mae,
        'r2': r2,
        'rmse_seconds': rmse_original,
        'mae_seconds': mae_original
    }


# ============================================================================
# 7. MODEL PERSISTENCE (SAVE/LOAD)
# ============================================================================

def save_model_artifacts(model, preprocessor, model_name, results, save_dir=None):
    """
    Save trained model, preprocessor, and results metadata to .pkl files.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model object
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    model_name : str
        Name of the model (e.g., 'Ridge')
    results : dict
        Model performance results/metrics
    save_dir : str, optional
        Directory to save files. If None, uses current directory
    """
    
    if save_dir is None:
        save_dir = os.getcwd()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING MODEL ARTIFACTS")
    print(f"{'='*70}")
    
    # Save model
    model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(save_dir, f'{model_name}_preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✓ Preprocessor saved: {preprocessor_path}")
    
    # Save results/metrics
    results_path = os.path.join(save_dir, f'{model_name}_results.pkl')
    joblib.dump(results, results_path)
    print(f"✓ Results saved: {results_path}")
    
    print(f"\nAll artifacts saved successfully to: {save_dir}")
    return model_path, preprocessor_path, results_path


def load_model_artifacts(model_name, load_dir=None):
    """
    Load trained model, preprocessor, and results from .pkl files.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'Ridge')
    load_dir : str, optional
        Directory containing saved files. If None, uses current directory
    
    Returns:
    --------
    model : sklearn model
        Trained model
    preprocessor : ColumnTransformer
        Fitted preprocessor
    results : dict
        Performance metrics
    """
    
    if load_dir is None:
        load_dir = os.getcwd()
    
    model_path = os.path.join(load_dir, f'{model_name}_model.pkl')
    preprocessor_path = os.path.join(load_dir, f'{model_name}_preprocessor.pkl')
    results_path = os.path.join(load_dir, f'{model_name}_results.pkl')
    
    print(f"Loading model artifacts from: {load_dir}")
    
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {model_path}")
    
    preprocessor = joblib.load(preprocessor_path)
    print(f"✓ Preprocessor loaded: {preprocessor_path}")
    
    results = joblib.load(results_path)
    print(f"✓ Results loaded: {results_path}")
    
    return model, preprocessor, results


# ============================================================================
# 8. PRODUCTION DEPLOYMENT WRAPPER
# ============================================================================

class TaxiDurationPredictor:
    """
    Production-ready wrapper for end-to-end taxi duration prediction.
    
    Handles:
    - Raw user input (pickup_datetime, lat/lon, etc.)
    - Feature engineering (all transformations)
    - Preprocessing (scaling, encoding)
    - Model prediction
    - Output interpretation (seconds to minutes/hours)
    
    Usage:
    ------
    # Load model
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge', model_dir)
    
    # Make predictions on raw data
    raw_data = pd.DataFrame({
        'pickup_datetime': ['2016-03-15 10:30:00'],
        'pickup_latitude': [40.7580],
        'pickup_longitude': [-73.9855],
        'dropoff_latitude': [40.7489],
        'dropoff_longitude': [-73.9680],
        'passenger_count': [1],
        'vendor_id': [2],
        'store_and_fwd_flag': ['N']
    })
    
    predictions = predictor.predict(raw_data)
    print(predictions)
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_cols = [
            'vendor_id', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
            'store_and_fwd_flag', 'hour', 'day_of_week', 'distance_km',
            'is_rush_hour', 'is_weekday', 'is_zero_distance', 'passenger_group',
            'distance_hour_interaction', 'distance_rush_interaction', 'lat_lon_distance',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'distance_log', 'distance_category'
        ]
        self.is_loaded = False
    
    def load_model(self, model_name='Ridge', model_dir=None):
        """Load saved model and preprocessor."""
        self.model, self.preprocessor, self.results = load_model_artifacts(model_name, model_dir)
        self.is_loaded = True
        print(f"✓ TaxiDurationPredictor ready for deployment")
    
    def _engineer_features(self, df):
        """Apply feature engineering (same as training)."""
        return engineer_features(df)
    
    def predict(self, raw_data):
        """
        Make predictions on raw user input.
        
        Parameters:
        -----------
        raw_data : pd.DataFrame
            Must contain columns: 
            - pickup_datetime (str or datetime)
            - pickup_latitude, pickup_longitude
            - dropoff_latitude, dropoff_longitude
            - passenger_count (int)
            - vendor_id (int: 1 or 2)
            - store_and_fwd_flag (str: 'Y' or 'N')
        
        Returns:
        --------
        dict with predictions
        """
        
        if not self.is_loaded:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        # Validate input
        required_cols = ['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                        'dropoff_latitude', 'dropoff_longitude', 'passenger_count',
                        'vendor_id', 'store_and_fwd_flag']
        
        missing = [col for col in required_cols if col not in raw_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Feature engineering
        engineered_data = self._engineer_features(raw_data.copy())
        
        # Extract features
        X = engineered_data[self.feature_cols].copy()
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        # Predict (in log space)
        y_pred_log = self.model.predict(X_processed)
        
        # Reverse log transformation
        y_pred_seconds = np.expm1(y_pred_log)
        
        # Clip to reasonable range (prevent outliers)
        y_pred_seconds = np.clip(y_pred_seconds, 60, 3600*2)  # 1 min to 2 hours
        
        # Create output
        results = pd.DataFrame({
            'predicted_duration_seconds': y_pred_seconds.astype(int),
            'predicted_duration_minutes': (y_pred_seconds / 60).astype(int),
            'predicted_duration_hours_minutes': [
                f"{int(sec//3600)}h {int((sec%3600)//60)}m"
                for sec in y_pred_seconds
            ]
        })
        
        # Add input features for transparency
        results['pickup_datetime'] = raw_data['pickup_datetime'].values
        results['passenger_count'] = raw_data['passenger_count'].values
        results['distance_km'] = engineered_data['distance_km'].values.round(2)
        
        return results
    
    def predict_single(self, pickup_dt, pickup_lat, pickup_lon, 
                       dropoff_lat, dropoff_lon, passenger_count, 
                       vendor_id, store_and_fwd_flag='N'):
        """
        Convenience method for single prediction.
        
        Returns: dict with prediction
        """
        data = pd.DataFrame({
            'pickup_datetime': [pickup_dt],
            'pickup_latitude': [pickup_lat],
            'pickup_longitude': [pickup_lon],
            'dropoff_latitude': [dropoff_lat],
            'dropoff_longitude': [dropoff_lon],
            'passenger_count': [passenger_count],
            'vendor_id': [vendor_id],
            'store_and_fwd_flag': [store_and_fwd_flag]
        })
        
        result = self.predict(data)
        return result.iloc[0].to_dict()




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
    print(f"Interaction features: distance_hour_interaction, distance_rush_interaction, lat_lon_distance")
    print(f"Cyclic time features: hour_sin, hour_cos, day_sin, day_cos")
    print(f"Distance features: distance_log, distance_category")

    # Separate features and target
    feature_cols = [
        'vendor_id', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
        'store_and_fwd_flag', 'hour', 'day_of_week', 'distance_km',
        'is_rush_hour', 'is_weekday', 'is_zero_distance', 'passenger_group',
        'distance_hour_interaction', 'distance_rush_interaction', 'lat_lon_distance',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'distance_log', 'distance_category'
    ]

    X_train = train_engineered[feature_cols].copy()
    y_train = train_engineered['trip_duration'].copy()

    X_val = val_engineered[feature_cols].copy()
    y_val = val_engineered['trip_duration'].copy()

    X_test = test_engineered[feature_cols].copy()
    y_test = test_engineered['trip_duration'].copy()

    # Log transform target (EDA: highly right-skewed, 5.09% outliers)
    print("\nApplying log transformation to target variable...")
    y_train_log = np.log1p(y_train)  # log1p = log(1 + x) handles zeros
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    print(f"Original target - mean: {y_train.mean():.2f}s, std: {y_train.std():.2f}s")
    print(f"Log target - mean: {y_train_log.mean():.4f}, std: {y_train_log.std():.4f}")

    # Create preprocessing pipeline
    print("\n" + "="*70)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*70)

    preprocessor, num_features, poly_features, cat_features, onehot_features = create_preprocessing_pipeline()
    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Polynomial features ({len(poly_features)}): {poly_features}")
    print(f"Categorical/Binary features ({len(cat_features)}): {cat_features}")
    print(f"One-hot features ({len(onehot_features)}): {onehot_features}")

    # Create models
    models = create_models()

    # Train and evaluate each model
    results = {}
    trained_models = {}

    for model_name, model in models.items():
        # Create new preprocessor for each model (fresh fit)
        preprocessor_copy, _, _, _, _ = create_preprocessing_pipeline()

        model_fitted, prep_fitted, y_train_pred, y_val_pred = train_model(
            X_train, y_train_log, X_val, y_val_log,
            preprocessor_copy, model_name, model
        )

        # Evaluate (pass y_train_log for proper clipping bounds)
        train_metrics = evaluate_model(y_train_log, y_train_pred, 'Training', y_train_log)
        val_metrics = evaluate_model(y_val_log, y_val_pred, 'Validation', y_train_log)

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
    print(f"\nBest Model: {best_model_name} (R2 = {results[best_model_name]['val']['r2']:.4f})")

    # Final test set evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION (Final Model)")
    print("="*70)

    best_model, best_preprocessor = trained_models[best_model_name]
    X_test_processed = best_preprocessor.transform(X_test)
    y_test_pred_log = best_model.predict(X_test_processed)

    # Clip predictions to reasonable bounds (based on training data range)
    # Log training mean ± 3 std to allow for reasonable variation
    log_mean = y_train_log.mean()
    log_std = y_train_log.std()
    min_bound = log_mean - 3 * log_std
    max_bound = log_mean + 3 * log_std
    y_test_pred_log = np.clip(y_test_pred_log, min_bound, max_bound)

    # Evaluate on test set
    test_metrics = evaluate_model(y_test_log, y_test_pred_log, 'Test', y_train_log)

    y_test_pred = np.expm1(y_test_pred_log)  # Reverse transformation

    print(f"\nTest predictions summary:")
    print(f"  Mean predicted duration: {y_test_pred.mean():.2f} seconds ({y_test_pred.mean()/60:.2f} minutes)")
    print(f"  Std predicted duration: {y_test_pred.std():.2f} seconds")
    print(f"  Min: {y_test_pred.min():.2f}s, Max: {y_test_pred.max():.2f}s")

    # Save model artifacts
    model_save_dir = os.path.dirname(os.path.abspath(__file__))
    save_model_artifacts(
        best_model, 
        best_preprocessor, 
        best_model_name,
        results[best_model_name],
        save_dir=model_save_dir
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    return results, trained_models, best_model_name, y_test_pred


if __name__ == '__main__':
    results, trained_models, best_model, predictions = main()
