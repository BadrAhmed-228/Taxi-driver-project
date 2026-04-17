import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
train_df = pd.read_csv('Project Data/train.csv')
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

# Extract features
train_df['hour'] = train_df['pickup_datetime'].dt.hour
train_df['day_of_week'] = train_df['pickup_datetime'].dt.day_name()

# Calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

train_df['distance_km'] = haversine_distance(
    train_df['pickup_latitude'], train_df['pickup_longitude'],
    train_df['dropoff_latitude'], train_df['dropoff_longitude']
)

# ============================================================================
# 1. TRIP DURATION DISTRIBUTION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Full distribution
axes[0, 0].hist(train_df['trip_duration'], bins=100, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Trip Duration (seconds)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trip Duration Distribution (Full Range)')
axes[0, 0].set_yscale('log')

# Filtered distribution (< 2 hours)
filtered = train_df[train_df['trip_duration'] < 7200]
axes[0, 1].hist(filtered['trip_duration'], bins=100, color='lightcoral', edgecolor='black')
axes[0, 1].set_xlabel('Trip Duration (seconds)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Trip Duration Distribution (< 2 hours)')

# In minutes
axes[1, 0].hist(train_df['trip_duration']/60, bins=100, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Trip Duration (minutes)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Trip Duration Distribution in Minutes')
axes[1, 0].set_xlim(0, 100)

# Box plot
axes[1, 1].boxplot(filtered['trip_duration']/60, vert=True)
axes[1, 1].set_ylabel('Trip Duration (minutes)')
axes[1, 1].set_title('Trip Duration Box Plot (< 2 hours)')
axes[1, 1].grid(axis='y')

plt.tight_layout()
plt.savefig('EDA_Visualizations/01_trip_duration_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[DONE] Saved: 01_trip_duration_distribution.png")

# ============================================================================
# 2. TIME-BASED PATTERNS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Trips by hour
hourly_stats = train_df.groupby('hour').size()
axes[0, 0].bar(hourly_stats.index, hourly_stats.values, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Number of Trips')
axes[0, 0].set_title('Trips by Hour of Day')
axes[0, 0].set_xticks(range(0, 24, 2))

# Avg duration by hour
hourly_duration = train_df.groupby('hour')['trip_duration'].mean() / 60
axes[0, 1].plot(hourly_duration.index, hourly_duration.values, marker='o', linewidth=2, markersize=6, color='darkred')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Avg Trip Duration (minutes)')
axes[0, 1].set_title('Average Trip Duration by Hour')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(0, 24, 2))

# Day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_trips = train_df['day_of_week'].value_counts().reindex(day_order)
axes[1, 0].bar(range(len(day_order)), daily_trips.values, color='teal', edgecolor='black')
axes[1, 0].set_xticks(range(len(day_order)))
axes[1, 0].set_xticklabels(day_order, rotation=45, ha='right')
axes[1, 0].set_ylabel('Number of Trips')
axes[1, 0].set_title('Trips by Day of Week')

# Avg duration by day
daily_duration = train_df.groupby('day_of_week')['trip_duration'].mean().reindex(day_order) / 60
axes[1, 1].bar(range(len(day_order)), daily_duration.values, color='orange', edgecolor='black')
axes[1, 1].set_xticks(range(len(day_order)))
axes[1, 1].set_xticklabels(day_order, rotation=45, ha='right')
axes[1, 1].set_ylabel('Avg Trip Duration (minutes)')
axes[1, 1].set_title('Average Trip Duration by Day of Week')

plt.tight_layout()
plt.savefig('EDA_Visualizations/02_temporal_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 02_temporal_patterns.png")

# ============================================================================
# 3. DISTANCE ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distance distribution
axes[0, 0].hist(train_df['distance_km'], bins=100, color='mediumpurple', edgecolor='black')
axes[0, 0].set_xlabel('Distance (km)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distance Distribution')
axes[0, 0].set_xlim(0, 50)

# Distance vs trip duration
sample_df = train_df.sample(n=10000, random_state=42)
axes[0, 1].scatter(sample_df['distance_km'], sample_df['trip_duration']/60, alpha=0.3, s=10, color='darkblue')
axes[0, 1].set_xlabel('Distance (km)')
axes[0, 1].set_ylabel('Trip Duration (minutes)')
axes[0, 1].set_title('Distance vs Trip Duration (Sample of 10k)')
axes[0, 1].set_xlim(0, 50)
axes[0, 1].set_ylim(0, 200)

# Distance bins
distance_bins = pd.cut(train_df['distance_km'], bins=[0, 1, 5, 10, 20, 1300],
                       labels=['<1km', '1-5km', '5-10km', '10-20km', '>20km'])
distance_counts = distance_bins.value_counts().sort_index()
axes[1, 0].bar(range(len(distance_counts)), distance_counts.values, color='mediumseagreen', edgecolor='black')
axes[1, 0].set_xticks(range(len(distance_counts)))
axes[1, 0].set_xticklabels(distance_counts.index)
axes[1, 0].set_ylabel('Number of Trips')
axes[1, 0].set_title('Trips by Distance Range')

# Avg duration by distance
distance_duration = train_df.groupby(distance_bins)['trip_duration'].mean() / 60
axes[1, 1].bar(range(len(distance_duration)), distance_duration.values, color='coral', edgecolor='black')
axes[1, 1].set_xticks(range(len(distance_duration)))
axes[1, 1].set_xticklabels(distance_duration.index)
axes[1, 1].set_ylabel('Avg Trip Duration (minutes)')
axes[1, 1].set_title('Average Trip Duration by Distance Range')

plt.tight_layout()
plt.savefig('EDA_Visualizations/03_distance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 03_distance_analysis.png")

# ============================================================================
# 4. PASSENGER COUNT ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Passenger count distribution
passenger_counts = train_df['passenger_count'].value_counts().sort_index()
axes[0, 0].bar(passenger_counts.index, passenger_counts.values, color='indianred', edgecolor='black')
axes[0, 0].set_xlabel('Passenger Count')
axes[0, 0].set_ylabel('Number of Trips')
axes[0, 0].set_title('Trips by Passenger Count')
axes[0, 0].set_yscale('log')

# Avg duration by passenger count
passenger_duration = train_df.groupby('passenger_count')['trip_duration'].mean() / 60
axes[0, 1].bar(passenger_duration.index, passenger_duration.values, color='lightblue', edgecolor='black')
axes[0, 1].set_xlabel('Passenger Count')
axes[0, 1].set_ylabel('Avg Trip Duration (minutes)')
axes[0, 1].set_title('Average Trip Duration by Passenger Count')
axes[0, 1].set_ylim(0, 25)

# Vendor comparison
vendor_passenger = train_df.groupby(['vendor_id', 'passenger_count']).size().unstack(fill_value=0)
vendor_passenger.T.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral'], edgecolor='black')
axes[1, 0].set_xlabel('Passenger Count')
axes[1, 0].set_ylabel('Number of Trips')
axes[1, 0].set_title('Vendor Distribution by Passenger Count')
axes[1, 0].legend(title='Vendor', labels=['Vendor 1', 'Vendor 2'])
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# Store and forward flag
sff_stats = train_df.groupby('store_and_fwd_flag')['trip_duration'].mean() / 60
axes[1, 1].bar(sff_stats.index, sff_stats.values, color=['green', 'red'], edgecolor='black')
axes[1, 1].set_xlabel('Store and Forward Flag')
axes[1, 1].set_ylabel('Avg Trip Duration (minutes)')
axes[1, 1].set_title('Average Trip Duration by Store and Forward Flag')
axes[1, 1].set_xticklabels(['No (N)', 'Yes (Y)'])

plt.tight_layout()
plt.savefig('EDA_Visualizations/04_passenger_and_vendor_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 04_passenger_and_vendor_analysis.png")

# ============================================================================
# 5. GEOGRAPHICAL ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Pickup locations
sample_df = train_df.sample(n=50000, random_state=42)
axes[0, 0].scatter(sample_df['pickup_longitude'], sample_df['pickup_latitude'],
                   c=sample_df['trip_duration']/60, cmap='viridis', s=1, alpha=0.5)
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')
axes[0, 0].set_title('Pickup Locations (Sample, colored by duration)')
cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
cbar.set_label('Trip Duration (min)')

# Dropoff locations
axes[0, 1].scatter(sample_df['dropoff_longitude'], sample_df['dropoff_latitude'],
                   c=sample_df['trip_duration']/60, cmap='viridis', s=1, alpha=0.5)
axes[0, 1].set_xlabel('Longitude')
axes[0, 1].set_ylabel('Latitude')
axes[0, 1].set_title('Dropoff Locations (Sample, colored by duration)')
cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
cbar.set_label('Trip Duration (min)')

# Pickup Longitude distribution
axes[1, 0].hist(train_df['pickup_longitude'], bins=100, color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Pickup Longitude')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Pickup Longitude Distribution')

# Pickup Latitude distribution
axes[1, 1].hist(train_df['pickup_latitude'], bins=100, color='lightcoral', edgecolor='black')
axes[1, 1].set_xlabel('Pickup Latitude')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Pickup Latitude Distribution')

plt.tight_layout()
plt.savefig('EDA_Visualizations/05_geographical_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 05_geographical_analysis.png")

# ============================================================================
# 6. CORRELATION HEATMAP
# ============================================================================
fig, axes = plt.subplots(figsize=(10, 8))

numeric_cols = ['passenger_count', 'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'trip_duration', 'distance_km', 'hour']

correlation_matrix = train_df[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            cbar_kws={'label': 'Correlation'}, square=True, ax=axes,
            cbar=True, linewidths=0.5, linecolor='gray')
axes.set_title('Correlation Heatmap of Numeric Features')

plt.tight_layout()
plt.savefig('EDA_Visualizations/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 06_correlation_heatmap.png")

# ============================================================================
# 7. VENDOR COMPARISON
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Trip count by vendor
vendor_counts = train_df['vendor_id'].value_counts().sort_index()
axes[0, 0].bar(vendor_counts.index, vendor_counts.values, color=['steelblue', 'coral'], edgecolor='black')
axes[0, 0].set_xlabel('Vendor ID')
axes[0, 0].set_ylabel('Number of Trips')
axes[0, 0].set_title('Trip Count by Vendor')
axes[0, 0].set_xticks([1, 2])

# Avg trip duration by vendor
vendor_duration = train_df.groupby('vendor_id')['trip_duration'].mean() / 60
axes[0, 1].bar(vendor_duration.index, vendor_duration.values, color=['steelblue', 'coral'], edgecolor='black')
axes[0, 1].set_xlabel('Vendor ID')
axes[0, 1].set_ylabel('Avg Trip Duration (minutes)')
axes[0, 1].set_title('Average Trip Duration by Vendor')
axes[0, 1].set_xticks([1, 2])

# Distance distribution by vendor
vendor1_dist = train_df[train_df['vendor_id'] == 1]['distance_km']
vendor2_dist = train_df[train_df['vendor_id'] == 2]['distance_km']
axes[1, 0].hist([vendor1_dist, vendor2_dist], bins=50, label=['Vendor 1', 'Vendor 2'],
                color=['steelblue', 'coral'], edgecolor='black')
axes[1, 0].set_xlabel('Distance (km)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distance Distribution by Vendor')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 25)

# Passenger count by vendor
vendor_passenger_mean = train_df.groupby('vendor_id')['passenger_count'].mean()
axes[1, 1].bar(vendor_passenger_mean.index, vendor_passenger_mean.values,
               color=['steelblue', 'coral'], edgecolor='black')
axes[1, 1].set_xlabel('Vendor ID')
axes[1, 1].set_ylabel('Avg Passenger Count')
axes[1, 1].set_title('Average Passenger Count by Vendor')
axes[1, 1].set_xticks([1, 2])
axes[1, 1].set_ylim(1, 2.5)

plt.tight_layout()
plt.savefig('EDA_Visualizations/07_vendor_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 07_vendor_comparison.png")

# ============================================================================
# 8. OUTLIER ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Trip duration outliers
Q1 = train_df['trip_duration'].quantile(0.25)
Q3 = train_df['trip_duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (train_df['trip_duration'] < lower_bound) | (train_df['trip_duration'] > upper_bound)
normal_data = train_df[~outlier_mask]['trip_duration']
outlier_data = train_df[outlier_mask]['trip_duration']

axes[0, 0].hist([normal_data/60, outlier_data/60], bins=50, label=['Normal', 'Outliers'],
                color=['skyblue', 'red'], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Trip Duration (minutes)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trip Duration: Normal vs Outliers')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 200)

# Distance outliers
Q1_dist = train_df['distance_km'].quantile(0.25)
Q3_dist = train_df['distance_km'].quantile(0.75)
IQR_dist = Q3_dist - Q1_dist
upper_bound_dist = Q3_dist + 1.5 * IQR_dist

outlier_mask_dist = train_df['distance_km'] > upper_bound_dist
normal_dist = train_df[~outlier_mask_dist]['distance_km']
outlier_dist = train_df[outlier_mask_dist]['distance_km']

axes[0, 1].hist([normal_dist, outlier_dist], bins=50, label=['Normal', 'Outliers'],
                color=['skyblue', 'red'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Distance (km)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distance: Normal vs Outliers')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 50)

# Outlier percentage by feature
outlier_pct = {
    'Trip Duration': len(outlier_data) / len(train_df) * 100,
    'Distance': len(outlier_dist) / len(train_df) * 100,
    'Zero Distance': (train_df['distance_km'] == 0).sum() / len(train_df) * 100,
    'Abnormal Speed': ((train_df['distance_km'] / (train_df['trip_duration'] / 3600)) > 100).sum() / len(train_df) * 100
}

axes[1, 0].bar(outlier_pct.keys(), outlier_pct.values(), color=['red', 'orange', 'yellow', 'pink'], edgecolor='black')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].set_title('Outlier Percentage by Feature')
axes[1, 0].tick_params(axis='x', rotation=45)

# Data quality summary
quality_issues = {
    'Zero Distance': (train_df['distance_km'] == 0).sum(),
    'Outside NYC': ((train_df['pickup_latitude'] < 40.5) | (train_df['pickup_latitude'] > 40.9) |
                   (train_df['pickup_longitude'] < -74.3) | (train_df['pickup_longitude'] > -73.7)).sum(),
    'Abnormal Speed': ((train_df['distance_km'] / (train_df['trip_duration'] / 3600)) > 100).sum()
}

axes[1, 1].bar(quality_issues.keys(), quality_issues.values(), color=['crimson', 'darkorange', 'gold'], edgecolor='black')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Data Quality Issues Count')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('EDA_Visualizations/08_outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Saved: 08_outlier_analysis.png")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
