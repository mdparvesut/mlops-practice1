import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_parquet('green_tripdata_2021-01.parquet')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nStatistics:")
print(df[['trip_distance', 'fare_amount', 'total_amount', 'passenger_count']].describe())
print("\nMissing values:")
print(df.isnull().sum())

# Clean data
df_clean = df[
    (df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
    (df['fare_amount'] > 0) & (df['fare_amount'] < 100)
].copy()
df_clean['pickup_hour'] = pd.to_datetime(df_clean['lpep_pickup_datetime']).dt.hour

print("\nCleaned shape:", df_clean.shape)

# Plot 1: Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_clean['trip_distance'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Trip Distance Distribution')
axes[0].set_xlabel('Miles')
axes[0].set_ylabel('Count')

axes[1].hist(df_clean['fare_amount'], bins=50, color='coral', edgecolor='white')
axes[1].set_title('Fare Amount Distribution')
axes[1].set_xlabel('Dollars ($)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('plot_distributions.png')
plt.show()
print("Saved: plot_distributions.png")

# Plot 2: Correlation heatmap
cols = ['trip_distance', 'fare_amount', 'total_amount', 'tip_amount', 'passenger_count']
plt.figure(figsize=(7, 5))
sns.heatmap(df_clean[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('plot_correlation.png')
plt.show()
print("Saved: plot_correlation.png")

# Plot 3: Trips by hour
plt.figure(figsize=(10, 4))
df_clean['pickup_hour'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Trips by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.tight_layout()
plt.savefig('plot_trips_by_hour.png')
plt.show()
print("Saved: plot_trips_by_hour.png")

print("\nEDA complete!")