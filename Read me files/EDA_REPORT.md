# INTENSIVE EXPLORATORY DATA ANALYSIS (EDA) REPORT
## Taxi Trip Duration Prediction Dataset

---

## EXECUTIVE SUMMARY

This report presents a comprehensive exploratory data analysis of a taxi trip duration prediction dataset containing **1,458,641 records** across training, validation, and test sets. The dataset spans 6 months (January-June 2016) of taxi rides in New York City.

### Key Findings:
- **1M training records** with detailed pickup/dropoff locations, passenger count, and trip duration
- **Clean dataset** with zero missing values
- **Moderate target variable skewness** - mean trip duration is 15.91 minutes, but highly variable
- **Strong geographic patterns** - Manhattan-based trips with specific density hotspots
- **Temporal variations** - rush hours (12-20) show longer trip durations
- **Multiple outliers** - 50,939 extreme duration cases warrant investigation

---

## 1. DATASET OVERVIEW

### Dataset Composition
| Metric | Value |
|--------|-------|
| Training Set | 1,000,000 records |
| Validation Set | 229,319 records |
| Test Set | 229,322 records |
| **Total Records** | **1,458,641** |
| Date Range | 2016-01-01 to 2016-06-30 (181 days) |

### Features (10 columns)
1. **id** - Unique identifier (object)
2. **vendor_id** - Taxi vendor (1 or 2) (int64)
3. **pickup_datetime** - Pickup timestamp (object)
4. **passenger_count** - Number of passengers (int64)
5. **pickup_longitude** - Pickup location longitude (float64)
6. **pickup_latitude** - Pickup location latitude (float64)
7. **dropoff_longitude** - Dropoff location longitude (float64)
8. **dropoff_latitude** - Dropoff location latitude (float64)
9. **store_and_fwd_flag** - Store and forward flag Y/N (object)
10. **trip_duration** - Duration in seconds [TARGET] (int64)

---

## 2. DATA QUALITY ASSESSMENT

### Missing Values
✓ **EXCELLENT**: Zero missing values in all 1,458,641 records
- Training: 0% missing
- Validation: 0% missing
- Test: 0% missing

### Data Quality Issues Identified
| Issue | Count | Percentage | Severity |
|-------|-------|-----------|----------|
| Zero distance trips | 4,085 | 0.41% | Low |
| Coordinates outside NYC | 222 | 0.02% | Low |
| Abnormal speed (>100 km/h) | 127 | 0.01% | Low |
| Trip duration outliers | 50,939 | 5.09% | Moderate |
| Distance outliers | 96,420 | 9.64% | Moderate |

---

## 3. TARGET VARIABLE ANALYSIS (Trip Duration)

### Statistical Summary
| Metric | Value |
|--------|-------|
| Count | 1,000,000 |
| Mean | 954.89 sec (15.91 min) |
| Median | 662 sec (11.03 min) |
| Std Dev | 3,882.07 sec |
| Min | 1 sec (0.02 min) |
| 25th percentile | 397 sec (6.62 min) |
| 75th percentile | 1,074 sec (17.9 min) |
| Max | 2,227,612 sec (37,127 min / 25.8 days) |

### Trip Duration Distribution
- **Very short trips** (< 1 min): 5,809 trips (0.58%)
- **Short trips** (1-5 min): 150,000+ trips
- **Normal trips** (5-30 min): 850,000+ trips
- **Long trips** (> 1 hour): 8,361 trips (0.84%)
- **Extreme outliers** (> 34.83 min): 50,939 trips (5.09%)

### Key Insight
The distribution is **highly right-skewed** with extreme outliers. Most trips are 5-30 minutes, but some trips exceed 24 hours, indicating potential data quality issues or special circumstances (e.g., stored rides).

---

## 4. CATEGORICAL FEATURES ANALYSIS

### 4.1 Vendor ID
| Vendor | Count | Percentage |
|--------|-------|-----------|
| Vendor 2 | 534,793 | 53.48% |
| Vendor 1 | 465,207 | 46.52% |

**Insights:**
- Balanced distribution between vendors
- Vendor 1: Avg 13.92 min trips, 3.40 km avg distance
- Vendor 2: Avg 17.65 min trips, 3.47 km avg distance
- **Vendor 2 has 27% longer trips on average**

### 4.2 Store and Forward Flag
| Flag | Count | Percentage |
|------|-------|-----------|
| N (No) | 994,443 | 99.44% |
| Y (Yes) | 5,557 | 0.56% |

**Insights:**
- Highly imbalanced: 99.4% of trips are normal uploads
- Store and forward (Y): Avg 17.93 min (13% longer)
- Indicates offline GPS logging in low-connectivity areas

### 4.3 Passenger Count
| Passengers | Count | Percentage |
|------------|-------|-----------|
| 0 | 38 | 0.004% |
| 1 | 708,628 | 70.86% |
| 2 | 143,907 | 14.39% |
| 3 | 41,012 | 4.10% |
| 4 | 19,491 | 1.95% |
| 5 | 53,635 | 5.36% |
| 6 | 33,287 | 3.33% |
| 7 | 2 | 0.0002% |

**Insights:**
- Dominated by single passengers (70.86%)
- Trip duration increases with passenger count:
  - 1 passenger: 15.42 min avg
  - 5 passengers: 17.87 min avg (16% increase)
- Groups (5-6 passengers) suggest special trips

---

## 5. TEMPORAL PATTERNS ANALYSIS

### 5.1 Hourly Patterns

**Busiest Hours:**
1. Hour 18 (6 PM): 62,028 trips
2. Hour 19 (7 PM): 61,701 trips
3. Hour 21 (9 PM): 57,790 trips

**Quietest Hours:**
- Hour 4 (4 AM): 10,853 trips
- Hour 5 (5 AM): 10,232 trips
- Hour 3 (3 AM): 14,360 trips

**Trip Duration by Hour:**
- Shortest: Hour 6 (6 AM) - 13.80 min avg
- Longest: Hour 15 (3 PM) - 18.39 min avg
- **Peak hour (18:00) duration: 16.23 min**

### 5.2 Daily Patterns

| Day | Trips | Avg Duration |
|-----|-------|-------------|
| Thursday | 149,942 | 16.74 min |
| Friday | 152,835 | 16.59 min |
| Wednesday | 144,165 | 16.31 min |
| Tuesday | 138,794 | 16.23 min |
| Saturday | 151,454 | 15.32 min |
| Sunday | 134,321 | 15.03 min |
| Monday | 128,489 | 15.00 min |

**Key Insight:**
- Weekday trips are 10-12% longer than weekend trips
- Thursday is busiest; Monday is quietest
- Work-week effect is significant

### 5.3 Monthly Patterns

| Month | Trips | Avg Duration |
|-------|-------|-------------|
| March | 175,745 | ~16 min |
| February | 163,721 | ~16 min |
| April | 172,014 | ~16 min |
| May | 170,499 | ~16 min |
| January | 157,629 | ~16 min |
| June | 160,392 | ~16 min |

**Insight:** Fairly consistent across months (~16 min), suggesting stable traffic patterns.

---

## 6. GEOGRAPHICAL ANALYSIS

### 6.1 Geographic Coverage

**Pickup Coordinates:**
- Latitude: 34.36° to 51.88° (Range: 17.52°)
- Longitude: -121.93° to -61.34° (Range: 60.59°)
- Mean center: (-73.97°, 40.75°) - Manhattan

**Dropoff Coordinates:**
- Latitude: 34.36° to 43.92° (Range: 9.56°)
- Longitude: -121.93° to -61.34° (Range: 60.59°)
- Mean center: (-73.97°, 40.75°) - Manhattan

### 6.2 Distance Analysis

**Calculated Haversine Distance:**

| Metric | Value |
|--------|-------|
| Mean | 3.44 km |
| Median | 2.09 km |
| Std Dev | 4.42 km |
| Min | 0.00 km |
| Max | 1,240.91 km |

**Distance Distribution:**
| Range | Trips | Percentage |
|-------|-------|-----------|
| < 1 km | 168,668 | 16.87% |
| 1-5 km | 651,615 | 65.16% |
| 5-10 km | 118,687 | 11.87% |
| 10-20 km | 44,713 | 4.47% |
| > 20 km | 16,317 | 1.63% |

**Key Insight:**
- **81% of trips are under 5 km** (short urban trips)
- Distance and duration correlation: **0.121** (weak positive)
- Very long distances (>20 km, 1.63%) likely cross-borough trips

---

## 7. FEATURE CORRELATIONS

### Correlation with Trip Duration (Target)

| Feature | Correlation | Strength |
|---------|-------------|----------|
| distance_km | 0.121 | Weak Positive |
| pickup_longitude | 0.035 | Very Weak |
| dropoff_longitude | 0.022 | Very Weak |
| passenger_count | 0.012 | Very Weak |
| hour | 0.003 | Very Weak |
| dropoff_latitude | -0.028 | Very Weak Negative |
| pickup_latitude | -0.034 | Very Weak Negative |

### Key Observations
- **Distance is the strongest predictor** (correlation 0.121)
- Geographic coordinates have minimal direct correlation
- Temporal features (hour) have negligible correlation
- **Feature engineering needed** for better predictive power

---

## 8. OUTLIER AND ANOMALY DETECTION

### 8.1 Trip Duration Outliers

**IQR Method:**
- Q1 (25th percentile): 6.62 min
- Q3 (75th percentile): 17.90 min
- IQR: 11.28 min
- Lower Bound: -10.31 min
- Upper Bound: 34.83 min
- **Outliers (> 34.83 min): 50,939 (5.09%)**

### 8.2 Distance Outliers

**IQR Method:**
- Q1: 1.23 km
- Q3: 3.87 km
- IQR: 2.64 km
- Upper Bound: 7.83 km
- **Outliers (> 7.83 km): 96,420 (9.64%)**

### 8.3 Data Anomalies

| Anomaly | Count | Notes |
|---------|-------|-------|
| Zero distance trips | 4,085 | Same pickup/dropoff location |
| Coordinates outside NYC | 222 | Data entry errors or long-distance |
| Speed > 100 km/h | 127 | Impossible in urban NYC |
| Extreme durations (> 24h) | 8,361 | Possible data logging errors |

---

## 9. STATISTICAL INSIGHTS

### 9.1 Vendor Comparison

| Metric | Vendor 1 | Vendor 2 | Difference |
|--------|----------|----------|-----------|
| Avg Trip Duration | 13.92 min | 17.65 min | +27% |
| Avg Distance | 3.40 km | 3.47 km | +2% |
| Avg Passengers | 1.26 | 2.02 | +60% |
| Market Share | 46.52% | 53.48% | - |

**Insight:** Vendor 2 has longer trips but more passengers, suggesting different service types or coverage areas.

### 9.2 Temporal Dynamics

- **Hourly variation**: ±33% from mean (13.80 min to 18.39 min)
- **Daily variation**: ±12% from mean (15.00 min to 16.74 min)
- **Rush hour effect**: Trips 32% longer at peak (3 PM) vs off-peak

### 9.3 Distance-Duration Relationship

- Weak correlation (0.121) suggests **traffic/congestion is a major factor**
- Short trips (< 1 km) show high variability in duration
- Long trips (> 20 km) more predictable and linear

---

## 10. KEY RECOMMENDATIONS FOR MODELING

### 10.1 Data Preprocessing
1. **Handle outliers**: Consider capping extreme durations (> 2 hours)
2. **Zero distance trips**: Separate handling or filtering
3. **Feature engineering**: 
   - Extract temporal features (hour, day, month, is_rush_hour)
   - Calculate actual road distance (not just Haversine)
   - Add weather/traffic proxy features

### 10.2 Feature Importance
1. **Distance** - Most predictive (correlation 0.121)
2. **Pickup location** - Geographic patterns evident
3. **Hour of day** - Rush hour effects visible
4. **Day of week** - Weekday/weekend differences
5. **Vendor** - 27% average duration difference

### 10.3 Model Considerations
- **Target scaling**: Use log transformation (highly skewed)
- **Outlier handling**: Robust regression or separate model for extreme cases
- **Class imbalance**: Normal vs. extreme trips
- **Geographic features**: Consider grid-based or cluster-based features

### 10.4 Expected Model Performance
- **Easy baseline**: Distance-only model (R² ~0.015)
- **Reasonable target**: 0.3-0.5 R² with good feature engineering
- **Challenge**: Weak feature correlations require advanced modeling

---

## 11. DATA LIMITATIONS & CAVEATS

1. **Geographic anomalies**: Some coordinates outside NYC bounds (222 cases)
2. **Extreme outliers**: Trips up to 25+ days suggest data logging issues
3. **Weak predictors**: Limited features for strong prediction
4. **Class imbalance**: Store-and-forward flag only 0.56% of data
5. **Temporal coverage**: Only 6 months of data; seasonal patterns unclear
6. **Missing features**: Weather, traffic conditions, special events not captured

---

## 12. SUMMARY STATISTICS TABLE

| Aspect | Metric | Value |
|--------|--------|-------|
| **Dataset** | Total Records | 1,458,641 |
| | Training | 1,000,000 |
| | Missing Values | 0% |
| **Target** | Mean Duration | 15.91 min |
| | Median Duration | 11.03 min |
| | Std Deviation | 64.70 min |
| | Duration Range | 1 sec - 25.8 days |
| **Geography** | Mean Distance | 3.44 km |
| | Geographic Range | ~18° lat × ~60° lon |
| | Median Distance | 2.09 km |
| **Temporal** | Date Range | 181 days (6 months) |
| | Busiest Hour | 18:00 (62,028 trips) |
| | Busiest Day | Thursday (149,942 trips) |
| **Categorical** | Vendors | 2 (46.5% / 53.5%) |
| | Passengers | 0-7 (70.9% are solo) |
| | Store & Forward | 99.4% Normal, 0.6% Stored |
| **Quality** | Outliers (Duration) | 50,939 (5.09%) |
| | Outliers (Distance) | 96,420 (9.64%) |
| | Data Anomalies | 4,434 (0.44%) |

---

## CONCLUSION

This NYC taxi dataset is **clean and well-structured** with comprehensive geographic and temporal information. The primary challenge for predictive modeling is the **weak correlation between available features and trip duration**, suggesting that external factors (traffic, weather, road conditions) play a dominant role. 

**Recommended approach:** Combine distance with advanced feature engineering (temporal patterns, traffic proxies, geographic clustering) and robust regression techniques to handle outliers effectively.

---

**Report Generated:** 2026-04-17  
**Analysis Tool:** Python (pandas, numpy, scipy)  
**Dataset:** NYC Taxi Trip Duration (Kaggle Competition Dataset)
