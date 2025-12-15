import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# =====================
# KONFIGURASI HALAMAN
# =====================
st.set_page_config(
    page_title="Customer Classification & Regression App",
    layout="centered"
)

st.title("📊 Customer Classification & Regression App")
st.markdown(
    "Aplikasi ini menggunakan **Ensemble Method (Random Forest)** "
    "untuk melakukan **klasifikasi subscription** dan "
    "**prediksi churn risk pelanggan**."
)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_customers_cleaned (1).csv")

df = load_data()

features = ["age", "income", "credit_score", "total_spent"]
X = df[features]

# =====================
# TRAIN MODEL (CACHED)
# =====================
@st.cache_resource
def train_model(df):
    # --- Klasifikasi ---
    y_class = df["subscription"]
    Xc_train, _, yc_train, _ = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )
    clf.fit(Xc_train, yc_train)

    # --- Regresi ---
    y_reg = df["churn_risk"]
    Xr_train, _, yr_train, _ = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )
    reg.fit(Xr_train, yr_train)

    return clf, reg

clf, reg = train_model(df)

# =====================
# INPUT USER
# =====================
st.header("Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umur", 18, 80, 30)
    income = st.number_input("Pendapatan", 0, 100_000_000, 5_000_000)

with col2:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    total_spent = st.number_input("Total Pengeluaran", 0, 50_000_000, 1_000_000)

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "total_spent": total_spent
}])

# =====================
# HASIL PREDIKSI
# =====================
st.header("📌 Hasil Prediksi")

status = clf.predict(input_df)[0]
prob = clf.predict_proba(input_df)[0][1]
churn = reg.predict(input_df)[0]

if status == 1:
    st.success("Status Subscription: **Berlangganan**")
else:
    st.warning("Status Subscription: **Tidak Berlangganan**")

st.write("Probabilitas Subscription:", round(prob, 2))
st.write("Prediksi Churn Risk:", round(churn, 2))

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption(
    "Customer ML App | Ensemble Method (Random Forest)"
)
