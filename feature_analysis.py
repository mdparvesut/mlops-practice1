import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_parquet('green_tripdata_2021-01.parquet')
df_clean = df[
    (df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
    (df['fare_amount'] > 0) & (df['fare_amount'] < 100)
].copy()
df_clean['pickup_hour'] = pd.to_datetime(df_clean['lpep_pickup_datetime']).dt.hour

# 3 features
F1 = ['trip_distance', 'pickup_hour', 'passenger_count']
d1 = df_clean[F1 + ['fare_amount']].dropna()
cv1 = cross_val_score(
    RandomForestRegressor(n_estimators=50, random_state=42),
    d1[F1], d1['fare_amount'], cv=5, scoring='r2'
)

# 4 features — adding trip_type
F2 = ['trip_distance', 'pickup_hour', 'passenger_count', 'trip_type']
d2 = df_clean[F2 + ['fare_amount']].dropna().copy()
d2['trip_type'] = d2['trip_type'].fillna(1)
cv2 = cross_val_score(
    RandomForestRegressor(n_estimators=50, random_state=42),
    d2[F2], d2['fare_amount'], cv=5, scoring='r2'
)

print("=== Feature Comparison ===")
print(f"3 features CV R²: {cv1.mean():.4f} ± {cv1.std():.4f}")
print(f"4 features CV R²: {cv2.mean():.4f} ± {cv2.std():.4f}")
print(f"Improvement:      {cv2.mean() - cv1.mean():.4f}")
