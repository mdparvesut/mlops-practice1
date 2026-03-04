import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and clean data
df = pd.read_parquet('green_tripdata_2021-01.parquet')
df_clean = df[
    (df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
    (df['fare_amount'] > 0) & (df['fare_amount'] < 100)
].copy()
df_clean['pickup_hour'] = pd.to_datetime(df_clean['lpep_pickup_datetime']).dt.hour

# Features and target
FEATURES = ['trip_distance', 'pickup_hour', 'passenger_count']
df_model = df_clean[FEATURES + ['fare_amount']].dropna()

X = df_model[FEATURES]
y = df_model['fare_amount']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=== Regression Results ===")
print(f"RMSE : {rmse:.4f}  (average error in dollars)")
print(f"MAE  : {mae:.4f}  (average absolute error)")
print(f"R²   : {r2:.4f}  (1.0 = perfect, 0 = terrible)")

cv = cross_val_score(
    RandomForestRegressor(n_estimators=50, random_state=42),
    X, y, cv=5, scoring='r2'
)
print(f"5-fold CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(6, 3))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importance - Regression')
plt.tight_layout()
plt.savefig('regression_importance.png')
plt.show()

# Save model
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved: regression_model.pkl")