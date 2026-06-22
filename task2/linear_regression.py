import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("student_scores.csv")

X = data[['Hours']]
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

study_hours = 7.5
predicted_score = model.predict(pd.DataFrame([[study_hours]], columns=['Hours']))

print("Predicted Score:", predicted_score[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Linear Regression")
plt.savefig("linear_regression_graph.png")
plt.show()