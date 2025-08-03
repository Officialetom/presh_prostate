import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import pickle
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fpdf import FPDF

# Constants
ADMIN_PASSWORD = "admin123"
MODEL_FILE = "model6.pkl"
DB_FILE = "predictions.db"

# Persistent database
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    psa REAL,
    prostate_volume REAL,
    family_history INTEGER,
    prediction TEXT
)
""")
conn.commit()

# Session states
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "model_data" not in st.session_state:
    st.session_state.model_data = None

def admin_login():
    st.title("Admin Login - Prostate Cancer Risk Prediction System")
    password = st.text_input("Enter Admin Password", type="password")
    if st.button("Login"):
        if password == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid password")

def logout_button():
    if st.sidebar.button("🔓 Logout"):
        st.session_state.authenticated = False
        st.session_state.model_data = None
        st.rerun()

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
            st.session_state.model_data = model_data
            return model_data
    return None

def train_model(data):
    st.info("Training model...")
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    # Display results
    st.subheader("Model Evaluation")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Classification Report:**")
    st.json(report)
    st.write("**Confusion Matrix:**")
    st.write(matrix)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, scaler), f)
    st.session_state.model_data = (model, scaler)
    st.success("Model trained and saved!")
    return model, scaler

def predict_risk(model, scaler, input_data):
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df.drop("name", axis=1))
    return model.predict(scaled)[0]

def save_prediction(record, result):
    cursor.execute("""
        INSERT INTO predictions (name, age, psa, prostate_volume, family_history, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (record["name"], record["age"], record["psa"], record["prostate_volume"],
          record["family_history"], result))
    conn.commit()

def prediction_form(model, scaler):
    st.header("📋 Patient Data Entry for Prediction")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 20, 100)
    psa = st.number_input("PSA Level", 0.0)
    prostate_volume = st.number_input("Prostate Volume", 10.0)
    family_history = st.selectbox("Family History", ["Yes", "No"])
    fh = 1 if family_history == "Yes" else 0

    if st.button("Predict"):
        data = {"name": name, "age": age, "psa": psa, "prostate_volume": prostate_volume, "family_history": fh}
        prediction = predict_risk(model, scaler, data)
        result = "Positive Risk" if prediction == 1 else "Low Risk"
        st.success(f"{name} has {result}")
        save_prediction(data, result)

def view_predictions():
    st.header("View Predictions")
    cursor.execute("SELECT name, age, psa, prostate_volume, family_history, prediction FROM predictions")
    rows = cursor.fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=["Name", "Age", "PSA", "Prostate Volume", "Family History", "Prediction"])
        st.dataframe(df)
        export_pdf(df)
    else:
        st.info("No predictions yet.")

def export_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Prostate Cancer Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for _, row in df.iterrows():
        line = f"Name: {row['Name']}, Age: {row['Age']}, PSA: {row['PSA']}, Volume: {row['Prostate Volume']}, FH: {row['Family History']}, Prediction: {row['Prediction']}"
        pdf.cell(200, 10, txt=line, ln=True)
    b = io.BytesIO(pdf.output(dest='S').encode('latin1'))
    st.download_button("Download PDF Report", b, file_name="predictions_report.pdf")

def main_app():
    logout_button()
    st.title("Prostate Cancer Risk Predictor")

    model_data = load_model()
    if not model_data:
        st.warning("Train a model with CSV data.")
        uploaded = st.file_uploader("Upload CSV with columns: age, psa, prostate_volume, family_history, target", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if "target" not in df.columns:
                st.error("Dataset must include a 'target' column.")
            else:
                model_data = train_model(df)

    if model_data:
        model, scaler = model_data
        tab1, tab2 = st.tabs(["🔍 Predict", "📂 Prediction History"])
        with tab1:
            prediction_form(model, scaler)
        with tab2:
            view_predictions()

# Launch app
if __name__ == "__main__":
    if not st.session_state.authenticated:
        admin_login()
    else:
        main_app()
