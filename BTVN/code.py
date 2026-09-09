import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Tạo dữ liệu giá nhà
data = {
    "DienTich": [30, 40, 50, 60, 70, 80, 90, 100],
    "PhongNgu": [1, 1, 2, 2, 2, 3, 3, 4],
    "Gia": [1.5, 2.0, 2.5, 3.0, 3.5, 4.2, 4.8, 5.5]
}

df = pd.DataFrame(data)


# 2. Chọn dữ liệu đầu vào và giá nhà
X = df[["DienTich", "PhongNgu"]]
y = df["Gia"]


# 3. Chia dữ liệu thành Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Tạo mô hình hồi quy tuyến tính
model = LinearRegression()


# 5. Huấn luyện mô hình
model.fit(X_train, y_train)


# 6. Dự đoán
y_pred = model.predict(X_test)


# 7. Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




st.title("🏠 Dự đoán giá nhà")

st.subheader("Dữ liệu")
st.dataframe(df)

st.subheader("Đánh giá mô hình")

st.write("MAE:", round(mae, 2))
st.write("R²:", round(r2, 2))


# 8. Nhập thông tin nhà
st.subheader("Dự đoán giá nhà mới")

dien_tich = st.number_input(
    "Diện tích (m²)",
    min_value=10,
    value=50
)

phong_ngu = st.number_input(
    "Số phòng ngủ",
    min_value=1,
    value=2
)


# 9. Dự đoán giá
if st.button("Dự đoán"):

    nha_moi = [[dien_tich, phong_ngu]]

    gia_du_doan = model.predict(nha_moi)

    st.success(
        f"Giá nhà dự đoán: {gia_du_doan[0]:.2f} tỷ đồng"
    )