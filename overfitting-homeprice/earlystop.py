import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


df = pd.read_csv("data.csv")


X = df[["DienTich", "PhongNgu", "PhongTam"]]


y = df["Gia"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(20, 10),
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=42
    )
)


model.fit(X_train, y_train)


train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

print("EARLY STOPPING")
print("-" * 35)
print("Train MSE:", round(train_mse, 4))
print("Test MSE :", round(test_mse, 4))