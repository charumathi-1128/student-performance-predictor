import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset.csv")

print("\nFirst Five Rows")
print(df.head())


print("\nStatistics")
print(df.describe())


X = df[['Hours']]
y = df['Score']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)



hours = [[7.5]]

prediction = model.predict(hours)

print("\nPredicted Score")

print(prediction[0])



plt.scatter(df['Hours'],
            df['Score'])

plt.plot(df['Hours'],
         model.predict(X),
         linewidth=2)

plt.xlabel("Hours Studied")

plt.ylabel("Exam Score")

plt.title("Student Performance Predictor")

plt.savefig("score_prediction.png")

plt.show()