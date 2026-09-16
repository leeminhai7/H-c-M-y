import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data.csv")

X = df[["DienTich", "PhongNgu", "PhongTam"]]
y = df["Gia"]

small = df.iloc[:10]

X_small = small[["DienTich", "PhongNgu", "PhongTam"]]
y_small = small["Gia"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

scaler_s = StandardScaler()

X_train_s = scaler_s.fit_transform(X_train_s)
X_test_s = scaler_s.transform(X_test_s)

model_s = LinearRegression()
model_s.fit(X_train_s, y_train_s)

mse_small = mean_squared_error(
    y_test_s,
    model_s.predict(X_test_s)
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

mse_more = mean_squared_error(
    y_test,
    model.predict(X_test)
)

print("10 mau:", round(mse_small, 4))
print("20 mau:", round(mse_more, 4))