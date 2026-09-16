import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data.csv")

X = df[["DienTich", "PhongNgu", "PhongTam"]]
y = df["Gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degree = 3

print("RIDGE")
print("-" * 40)

for alpha in [0.01, 0.1, 1, 10, 100]:

    model = make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    model.fit(X_train, y_train)

    train_mse = mean_squared_error(
        y_train,
        model.predict(X_train)
    )

    test_mse = mean_squared_error(
        y_test,
        model.predict(X_test)
    )

    print(
        "alpha =", alpha,
        "| Train =", round(train_mse, 4),
        "| Test =", round(test_mse, 4)
    )

print()
print("LASSO")
print("-" * 40)

for alpha in [0.01, 0.1, 1, 10]:

    model = make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=10000)
    )

    model.fit(X_train, y_train)

    train_mse = mean_squared_error(
        y_train,
        model.predict(X_train)
    )

    test_mse = mean_squared_error(
        y_test,
        model.predict(X_test)
    )

    print(
        "alpha =", alpha,
        "| Train =", round(train_mse, 4),
        "| Test =", round(test_mse, 4)
    )