import os
import joblib
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ridge_pipeline.pkl")

def load_model():
    """Carica e ritorna il modello, oppure None se non esiste."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def predict_one(payload: dict):
    """Predice un singolo esempio (dict) ritornando un float."""
    model = load_model()
    if model is None:
        raise RuntimeError("Model file not found")
    df = pd.DataFrame([payload])
    return float(model.predict(df)[0])

def main():
    st.title("Airbnb Italy Price Prediction")
    model = load_model()
    st.sidebar.header("Input dei dati")
    uploaded_file = st.sidebar.file_uploader("Carica un CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        preds = model.predict(df) if model else ["no model"]
        df["predicted_price"] = preds
        st.write(df)
    else:
        st.sidebar.subheader("Inserimento manuale")
        room_type = st.sidebar.selectbox("Tipo di stanza", ["Entire home/apt", "Private room", "Shared room"])
        number_of_reviews = st.sidebar.number_input("Numero di recensioni", 0, 10000, 0)
        availability_365 = st.sidebar.number_input("Disponibilità annuale", 0, 365, 0)
        if st.sidebar.button("Predici prezzo"):
            payload = {
                "room_type": room_type,
                "number_of_reviews": number_of_reviews,
                "availability_365": availability_365
            }
            try:
                pred = predict_one(payload)
                st.write(f"€ {pred:.2f}")
            except RuntimeError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
