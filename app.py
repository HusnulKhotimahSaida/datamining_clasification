import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error

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
    "**regresi churn risk pelanggan**."
)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_customers_cleaned (1).csv")

df = load_data()

features = ["age", "income", "credit_score", "total_spent"]

# =====================
# INPUT USER
# =====================
st.header("🧑 Input Data Pelanggan")

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
# PROSES MODEL
# =====================
st.header("⚙️ Proses Model")

if st.button("🚀 Jalankan Model"):
    with st.spinner("Melatih model, mohon tunggu..."):

        X = df[features]

        # ===== KLASIFIKASI =====
        y_class = df["subscription"]
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(
            n_estimators=10,
            random_state=42
        )
        clf.fit(Xc_train, yc_train)

        # ===== REGRESI =====
        y_reg = df["churn_risk"]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

        reg = RandomForestRegressor(
            n_estimators=10,
            random_state=42
        )
        reg.fit(Xr_train, yr_train)

    st.success("Model berhasil dijalankan!")

    # =====================
    # HASIL PREDIKSI USER
    # =====================
    st.subheader("📌 Hasil Prediksi")

    status = clf.predict(input_df)[0]
    prob = clf.predict_proba(input_df)[0][1]
    churn = reg.predict(input_df)[0]

    st.write(
        "Status Subscription:",
        "Berlangganan" if status == 1 else "Tidak Berlangganan"
    )
    st.write("Probabilitas Subscription:", round(prob, 2))
    st.write("Prediksi Churn Risk:", round(churn, 2))

    # =====================
    # EVALUASI KLASIFIKASI
    # =====================
    st.header("📈 Evaluasi Model")

    st.subheader("🔵 Evaluasi Klasifikasi (Confusion Matrix)")
    cm = confusion_matrix(yc_test, clf.predict(Xc_test))

    fig1, ax1 = plt.subplots()
    ax1.imshow(cm)
    ax1.set_title("Confusion Matrix - Subscription")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    st.write("Accuracy:", accuracy_score(yc_test, clf.predict(Xc_test)))

    # =====================
    # EVALUASI REGRESI
    # =====================
    st.subheader("🟢 Evaluasi Regresi Churn Risk")

    y_pred_r = reg.predict(Xr_test)
    st.write("MAE (Mean Absolute Error):", round(
        mean_absolute_error(yr_test, y_pred_r), 3
    ))

    # =====================
    # VISUALISASI REGRESI (DIAGRAM GARIS)
    # =====================
    st.subheader("📉 Visualisasi Regresi (Diagram Garis)")

    # Ambil sebagian data agar grafik jelas
    sample_size = 50
    y_actual = yr_test.iloc[:sample_size].reset_index(drop=True)
    y_predicted = pd.Series(y_pred_r[:sample_size])

    fig2, ax2 = plt.subplots()
    ax2.plot(y_actual, label="Actual Churn Risk", marker="o")
    ax2.plot(y_predicted, label="Predicted Churn Risk", marker="x")
    ax2.set_xlabel("Data ke-")
    ax2.set_ylabel("Churn Risk")
    ax2.set_title("Perbandingan Actual vs Predicted Churn Risk")
    ax2.legend()
    st.pyplot(fig2)

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption(
    "Customer ML App | Ensemble Method (Random Forest) | Streamlit Cloud"
)
