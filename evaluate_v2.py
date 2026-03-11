import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_jan = pd.read_parquet('data/green_tripdata_2021-01.parquet')
df_feb = pd.read_parquet('data/green_tripdata_2021-02.parquet')

print("January rows:", len(df_jan))
print("February rows:", len(df_feb))

df = pd.concat([df_jan, df_feb], ignore_index=True)
print("Combined rows:", len(df))

df = df[
    (df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
    (df['fare_amount'] > 0) & (df['fare_amount'] < 100)
].copy()
df['pickup_hour'] = pd.to_datetime(df['lpep_pickup_datetime']).dt.hour

FEATURES = ['trip_distance', 'pickup_hour', 'passenger_count']
df_model = df[FEATURES + ['fare_amount']].dropna()

X = df_model[FEATURES]
y = df_model['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open('regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n=== V2: Old Model Tested on Combined Data ===")
print("RMSE :", round(rmse, 4))
print("MAE  :", round(mae, 4))
print("R2   :", round(r2, 4))
print("\nModel was trained on January only.")
print("This shows how well it works on February data.")