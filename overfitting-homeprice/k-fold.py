import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data.csv")

X = df[["DienTich", "PhongNgu", "PhongTam"]]
y = df["Gia"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Bac | Train MSE | Validation MSE")
print("-" * 35)

scores = []

for degree in range(1, 5):
    train_errors = []
    validation_errors = []

    for train_index, val_index in kf.split(X):
        X_train = X.iloc[train_index]
        X_val = X.iloc[val_index]

        y_train = y.iloc[train_index]
        y_val = y.iloc[val_index]

        model = make_pipeline(
            PolynomialFeatures(degree),
            StandardScaler(),
            LinearRegression()
        )

        model.fit(X_train, y_train)

        train_errors.append(
            mean_squared_error(y_train, model.predict(X_train))
        )

        validation_errors.append(
            mean_squared_error(y_val, model.predict(X_val))
        )

    train_mse = np.mean(train_errors)
    validation_mse = np.mean(validation_errors)

    scores.append(validation_mse)

    print(
        degree,
        "|",
        round(train_mse, 4),
        "|",
        round(validation_mse, 4)
    )

print("-" * 35)
print("Bac toi uu:", np.argmin(scores) + 1)