import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
train_df = pd.read_csv('Project Data/train.csv')
test_df = pd.read_csv('Project Data/test.csv')
val_df = pd.read_csv('Project Data/val.csv')

print("="*80)
print("INTENSIVE EXPLORATORY DATA ANALYSIS - TAXI TRIP DURATION PREDICTION")
print("="*80)

# ============================================================================
# 1. DATASET OVERVIEW
# ============================================================================
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Val set shape: {val_df.shape}")
print(f"Total records: {train_df.shape[0] + test_df.shape[0] + val_df.shape[0]}")

print("\nColumn names and types:")
print(train_df.dtypes)

print("\nFirst few rows:")
print(train_df.head())

# ============================================================================
# 2. MISSING VALUES ANALYSIS
# ============================================================================
print("\n2. MISSING VALUES ANALYSIS")
print("-" * 80)
print("Train set missing values:")
print(train_df.isnull().sum())
print(f"Missing percentage: {(train_df.isnull().sum() / len(train_df) * 100).round(2)}")

print("\nTest set missing values:")
print(test_df.isnull().sum())

print("\nVal set missing values:")
print(val_df.isnull().sum())

# ============================================================================
# 3. DATA TYPES & BASIC STATISTICS
# ============================================================================
print("\n3. BASIC STATISTICS")
print("-" * 80)
print(train_df.describe())

print("\nAdditional statistics:")
print(train_df[['passenger_count', 'pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude', 'trip_duration']].describe())

# ============================================================================
# 4. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n4. CATEGORICAL FEATURES ANALYSIS")
print("-" * 80)

print("\nVendor ID distribution:")
print(train_df['vendor_id'].value_counts())
print(f"Percentage: {train_df['vendor_id'].value_counts(normalize=True) * 100}")

print("\nStore and Forward Flag distribution:")
print(train_df['store_and_fwd_flag'].value_counts())
print(f"Percentage: {train_df['store_and_fwd_flag'].value_counts(normalize=True) * 100}")

print("\nPassenger Count distribution:")
print(train_df['passenger_count'].value_counts().sort_index())
print(f"Min: {train_df['passenger_count'].min()}, Max: {train_df['passenger_count'].max()}")
print(f"Unusual values: {train_df['passenger_count'].value_counts().tail(10)}")

# ============================================================================
# 5. TRIP DURATION ANALYSIS (TARGET VARIABLE)
# ============================================================================
print("\n5. TRIP DURATION ANALYSIS (TARGET VARIABLE)")
print("-" * 80)

trip_duration_stats = train_df['trip_duration'].describe()
print(trip_duration_stats)

print(f"\nTrip Duration in minutes - Min: {train_df['trip_duration'].min()/60:.2f}, "
      f"Max: {train_df['trip_duration'].max()/60:.2f}, "
      f"Mean: {train_df['trip_duration'].mean()/60:.2f}, "
      f"Median: {train_df['trip_duration'].median()/60:.2f}")

print(f"\nOutliers (> 3 std from mean): {len(train_df[train_df['trip_duration'] > train_df['trip_duration'].mean() + 3*train_df['trip_duration'].std()])}")
print(f"Very short trips (< 60 sec): {len(train_df[train_df['trip_duration'] < 60])}")
print(f"Very long trips (> 3600 sec): {len(train_df[train_df['trip_duration'] > 3600])}")

# ============================================================================
# 6. DATETIME ANALYSIS
# ============================================================================
print("\n6. DATETIME ANALYSIS")
print("-" * 80)

# Parse datetime
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

print(f"Date range: {train_df['pickup_datetime'].min()} to {train_df['pickup_datetime'].max()}")
print(f"Duration: {train_df['pickup_datetime'].max() - train_df['pickup_datetime'].min()}")

# Extract temporal features
train_df['hour'] = train_df['pickup_datetime'].dt.hour
train_df['day_of_week'] = train_df['pickup_datetime'].dt.day_name()
train_df['month'] = train_df['pickup_datetime'].dt.month
train_df['day'] = train_df['pickup_datetime'].dt.day

print("\nTrips by hour of day:")
print(train_df['hour'].value_counts().sort_index())

print("\nTrips by day of week:")
print(train_df['day_of_week'].value_counts())

print("\nTrips by month:")
print(train_df['month'].value_counts().sort_index())

# ============================================================================
# 7. GEOGRAPHICAL ANALYSIS
# ============================================================================
print("\n7. GEOGRAPHICAL ANALYSIS")
print("-" * 80)

print("\nPickup Longitude - Range and stats:")
print(f"Min: {train_df['pickup_longitude'].min()}, Max: {train_df['pickup_longitude'].max()}")
print(f"Mean: {train_df['pickup_longitude'].mean()}, Std: {train_df['pickup_longitude'].std()}")

print("\nPickup Latitude - Range and stats:")
print(f"Min: {train_df['pickup_latitude'].min()}, Max: {train_df['pickup_latitude'].max()}")
print(f"Mean: {train_df['pickup_latitude'].mean()}, Std: {train_df['pickup_latitude'].std()}")

print("\nDropoff Longitude - Range and stats:")
print(f"Min: {train_df['dropoff_longitude'].min()}, Max: {train_df['dropoff_longitude'].max()}")
print(f"Mean: {train_df['dropoff_longitude'].mean()}, Std: {train_df['dropoff_longitude'].std()}")

print("\nDropoff Latitude - Range and stats:")
print(f"Min: {train_df['dropoff_latitude'].min()}, Max: {train_df['dropoff_latitude'].max()}")
print(f"Mean: {train_df['dropoff_latitude'].mean()}, Std: {train_df['dropoff_latitude'].std()}")

# Calculate distance (Haversine formula)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

train_df['distance_km'] = haversine_distance(
    train_df['pickup_latitude'], train_df['pickup_longitude'],
    train_df['dropoff_latitude'], train_df['dropoff_longitude']
)

print("\nCalculated Distance (Haversine) - km:")
print(train_df['distance_km'].describe())

# ============================================================================
# 8. CORRELATIONS
# ============================================================================
print("\n8. CORRELATION ANALYSIS")
print("-" * 80)

numeric_cols = ['passenger_count', 'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'trip_duration',
                'distance_km', 'hour']

correlation_matrix = train_df[numeric_cols].corr()

print("\nCorrelation with trip_duration:")
print(correlation_matrix['trip_duration'].sort_values(ascending=False))

# ============================================================================
# 9. VENDOR-SPECIFIC ANALYSIS
# ============================================================================
print("\n9. VENDOR-SPECIFIC ANALYSIS")
print("-" * 80)

for vendor in sorted(train_df['vendor_id'].unique()):
    vendor_data = train_df[train_df['vendor_id'] == vendor]
    print(f"\nVendor {vendor}:")
    print(f"  Records: {len(vendor_data)}")
    print(f"  Avg trip duration: {vendor_data['trip_duration'].mean()/60:.2f} min")
    print(f"  Median trip duration: {vendor_data['trip_duration'].median()/60:.2f} min")
    print(f"  Avg distance: {vendor_data['distance_km'].mean():.2f} km")
    print(f"  Avg passengers: {vendor_data['passenger_count'].mean():.2f}")

# ============================================================================
# 10. TIME-BASED PATTERNS
# ============================================================================
print("\n10. TIME-BASED PATTERNS")
print("-" * 80)

print("\nAverage trip duration by hour:")
hourly_avg = train_df.groupby('hour')['trip_duration'].agg(['mean', 'median', 'count'])
hourly_avg['mean_min'] = hourly_avg['mean'] / 60
hourly_avg['median_min'] = hourly_avg['median'] / 60
print(hourly_avg[['mean_min', 'median_min', 'count']])

print("\nAverage trip duration by day of week:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_avg = train_df.groupby('day_of_week')['trip_duration'].agg(['mean', 'median', 'count'])
daily_avg = daily_avg.reindex([d for d in day_order if d in daily_avg.index])
daily_avg['mean_min'] = daily_avg['mean'] / 60
daily_avg['median_min'] = daily_avg['median'] / 60
print(daily_avg[['mean_min', 'median_min', 'count']])

# ============================================================================
# 11. PASSENGER COUNT ANALYSIS
# ============================================================================
print("\n11. PASSENGER COUNT PATTERNS")
print("-" * 80)

print("\nTrip duration by passenger count:")
passenger_analysis = train_df.groupby('passenger_count')['trip_duration'].agg(['mean', 'median', 'std', 'count'])
passenger_analysis['mean_min'] = passenger_analysis['mean'] / 60
passenger_analysis['median_min'] = passenger_analysis['median'] / 60
print(passenger_analysis)

# ============================================================================
# 12. DISTANCE ANALYSIS
# ============================================================================
print("\n12. DISTANCE ANALYSIS")
print("-" * 80)

print(f"Trips with distance < 1 km: {len(train_df[train_df['distance_km'] < 1])}")
print(f"Trips with distance 1-5 km: {len(train_df[(train_df['distance_km'] >= 1) & (train_df['distance_km'] < 5)])}")
print(f"Trips with distance 5-10 km: {len(train_df[(train_df['distance_km'] >= 5) & (train_df['distance_km'] < 10)])}")
print(f"Trips with distance 10-20 km: {len(train_df[(train_df['distance_km'] >= 10) & (train_df['distance_km'] < 20)])}")
print(f"Trips with distance > 20 km: {len(train_df[train_df['distance_km'] > 20])}")

print("\nDistance vs Trip Duration correlation:")
print(f"Correlation: {train_df['distance_km'].corr(train_df['trip_duration']):.4f}")

# ============================================================================
# 13. STORE AND FORWARD FLAG ANALYSIS
# ============================================================================
print("\n13. STORE AND FORWARD FLAG ANALYSIS")
print("-" * 80)

for flag in ['N', 'Y']:
    flag_data = train_df[train_df['store_and_fwd_flag'] == flag]
    print(f"\nStore and Forward: {flag}")
    print(f"  Records: {len(flag_data)}")
    print(f"  Avg trip duration: {flag_data['trip_duration'].mean()/60:.2f} min")
    print(f"  Median trip duration: {flag_data['trip_duration'].median()/60:.2f} min")

# ============================================================================
# 14. OUTLIER DETECTION
# ============================================================================
print("\n14. OUTLIER DETECTION")
print("-" * 80)

# Trip duration outliers
Q1 = train_df['trip_duration'].quantile(0.25)
Q3 = train_df['trip_duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nTrip Duration IQR Analysis:")
print(f"  Q1: {Q1/60:.2f} min, Q3: {Q3/60:.2f} min, IQR: {IQR/60:.2f} min")
print(f"  Lower bound: {lower_bound/60:.2f} min, Upper bound: {upper_bound/60:.2f} min")
print(f"  Outliers below lower bound: {len(train_df[train_df['trip_duration'] < lower_bound])}")
print(f"  Outliers above upper bound: {len(train_df[train_df['trip_duration'] > upper_bound])}")

# Distance outliers
Q1_dist = train_df['distance_km'].quantile(0.25)
Q3_dist = train_df['distance_km'].quantile(0.75)
IQR_dist = Q3_dist - Q1_dist
upper_bound_dist = Q3_dist + 1.5 * IQR_dist

print(f"\nDistance IQR Analysis:")
print(f"  Q1: {Q1_dist:.2f} km, Q3: {Q3_dist:.2f} km")
print(f"  Upper bound for outliers: {upper_bound_dist:.2f} km")
print(f"  Outliers: {len(train_df[train_df['distance_km'] > upper_bound_dist])}")

# ============================================================================
# 15. DATA QUALITY ISSUES
# ============================================================================
print("\n15. DATA QUALITY ISSUES")
print("-" * 80)

# Zero distance trips
zero_dist = len(train_df[train_df['distance_km'] == 0])
print(f"\nZero distance trips: {zero_dist}")

# Impossible coordinates (outside NYC area)
# NYC approximately: lat 40.5-40.9, lon -74.3 to -73.7
outside_nyc = len(train_df[
    (train_df['pickup_latitude'] < 40.5) | (train_df['pickup_latitude'] > 40.9) |
    (train_df['pickup_longitude'] < -74.3) | (train_df['pickup_longitude'] > -73.7)
])
print(f"Trips with coordinates outside NYC bounds: {outside_nyc}")

# Trips faster than physically possible (100+ km/h average speed)
train_df['avg_speed_kmh'] = (train_df['distance_km'] / (train_df['trip_duration'] / 3600)).fillna(0)
fast_trips = len(train_df[train_df['avg_speed_kmh'] > 100])
print(f"Trips with average speed > 100 km/h: {fast_trips}")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
