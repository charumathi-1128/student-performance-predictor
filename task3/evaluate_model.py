import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
data = pd.read_csv(os.path.join(BASE_DIR, "task2", "student_scores.csv"))

X = data[['Hours']]
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load saved model
model = joblib.load(os.path.join(BASE_DIR, "task2", "student_model.pkl"))

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted")
print(results)

# Predict for 4.5 hours
new_hours = pd.DataFrame({"Hours": [4.5]})
prediction = model.predict(new_hours)

print(f"\nPredicted score for 4.5 study hours: {prediction[0]:.2f}")