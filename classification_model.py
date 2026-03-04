import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)

# Load and clean data
df = pd.read_parquet('green_tripdata_2021-01.parquet')
df_clean = df[
    (df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
    (df['fare_amount'] > 0) & (df['fare_amount'] < 100)
].copy()
df_clean['pickup_hour'] = pd.to_datetime(df_clean['lpep_pickup_datetime']).dt.hour

# Features and target (did passenger tip? 1=yes, 0=no)
FEATURES = ['trip_distance', 'pickup_hour', 'passenger_count']
df_model = df_clean[FEATURES + ['tip_amount']].dropna().copy()
df_model['tip_given'] = (df_model['tip_amount'] > 0).astype(int)

print("Class split:")
print(df_model['tip_given'].value_counts())

X = df_model[FEATURES]
y = df_model['tip_given']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]

# Metrics
print("\n=== Classification Results ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

cv = cross_val_score(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y, cv=5, scoring='accuracy'
)
print(f"5-fold CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tip','Tip'], yticklabels=['No Tip','Tip'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('clf_confusion_matrix.png')
plt.show()

# Save model
with open('classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved: classification_model.pkl")