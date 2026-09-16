import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data.csv")

X = df[["DienTich", "PhongNgu", "PhongTam"]]
y = df["Gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Bac | Train MSE | Test MSE")
print("-" * 30)

for degree in range(1, 5):

    model = make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        LinearRegression()
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
        degree,
        "|",
        round(train_mse, 4),
        "|",
        round(test_mse, 4)
    )