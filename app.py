# prostate_cancer_app.py

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
MODEL_FILE = "model.pkl"
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

# Session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Session state for trained model
if "model_data" not in st.session_state:
    st.session_state.model_data = None

def admin_login():
    st.title("Admin Login - Prostate Cancer Risk Prediction System")
    password = st.text_input("Enter Admin Password", type="password")
    if st.button("Login"):
        if password == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful! Please wait...")
            st.rerun()
        else:
            st.error("Invalid password")

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
            st.session_state.model_data = model_data
            return model_data
    return None

def train_model(data):
    st.info("Training model...")

    # Drop non-numeric fields
    data = data.drop(columns=[col for col in data.columns if col.lower() in ["name", "id"]], errors='ignore')

    X = data.drop("target", axis=1)
    y = data["target"]

    # ✅ Ensure balanced splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    # Display results
    st.subheader("📊 Model Evaluation Results")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Classification Report:**")
    st.json(report)
    st.write("**Confusion Matrix:**")
    st.write(matrix)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, scaler), f)

    st.session_state.model_data = (model, scaler)
    st.success("Model trained and saved successfully!")
    return model, scaler

def predict_risk(model, scaler, input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df.drop("name", axis=1))
    prediction = model.predict(input_scaled)[0]
    return prediction

def prediction_form(model, scaler):
    st.header("📋 Patient Data Entry for Prediction")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=20, max_value=100)
    psa = st.number_input("PSA Level", min_value=0.0)
    prostate_volume = st.number_input("Prostate Volume", min_value=10.0)
    family_history = st.selectbox("Family History of Prostate Cancer", ["Yes", "No"])

    input_data = {
        "name": name,
        "age": age,
        "psa": psa,
        "prostate_volume": prostate_volume,
        "family_history": 1 if family_history == "Yes" else 0
    }

    if st.button("Predict"):
        prediction = predict_risk(model, scaler, input_data)
        result = "Positive Risk" if prediction == 1 else "Low Risk"
        st.success(f"Prediction for {name}: {result}")
        save_prediction(input_data, result)

def save_prediction(record, result):
    cursor.execute("""
        INSERT INTO predictions (name, age, psa, prostate_volume, family_history, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (record["name"], record["age"], record["psa"], record["prostate_volume"], record["family_history"], result))
    conn.commit()

def export_predictions_pdf():
    cursor.execute("SELECT name, age, psa, prostate_volume, family_history, prediction FROM predictions")
    rows = cursor.fetchall()
    if not rows:
        st.warning("No predictions to export.")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Prostate Cancer Predictions Report", ln=True, align="C")
    pdf.ln(10)

    for row in rows:
        name, age, psa, volume, history, prediction = row
        line = f"Name: {name}, Age: {age}, PSA: {psa}, Volume: {volume}, Family History: {history}, Prediction: {prediction}"
        pdf.cell(200, 10, txt=line, ln=True)

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output = io.BytesIO(pdf_bytes)
    st.download_button("Download PDF Report", pdf_output, file_name="predictions_report.pdf")

def view_predictions():
    st.header("📁 Prediction History")
    cursor.execute("SELECT name, age, psa, prostate_volume, family_history, prediction FROM predictions")
    rows = cursor.fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=["name", "age", "psa", "prostate_volume", "family_history", "prediction"])
        st.dataframe(df)
        export_predictions_pdf()
    else:
        st.info("No predictions made yet.")

def logout_button():
    if st.sidebar.button("🔓 Logout"):
        st.session_state.authenticated = False
        st.session_state.model_data = None
        st.rerun()

def main_app():
    logout_button()
    st.title("Prostate Cancer Risk Predictor")

    model_data = load_model()
    if not model_data:
        st.warning("No trained model found. Please upload a dataset to train the model.")
        uploaded = st.file_uploader("Upload CSV (must include 'target' column: 1 = Positive Risk, 0 = Low Risk)", type="csv")
        if uploaded:
            data = pd.read_csv(uploaded)
            if "target" not in data.columns:
                st.error("Dataset must include a 'target' column.")
            else:
                model_data = train_model(data)

    if model_data:
        model, scaler = model_data
        tab1, tab2 = st.tabs(["Predict", "View Predictions"])
        with tab1:
            prediction_form(model, scaler)
        with tab2:
            view_predictions()

if __name__ == "__main__":
    if not st.session_state.authenticated:
        admin_login()
    else:
        main_app()
