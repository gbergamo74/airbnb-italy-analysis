import os
import joblib
import streamlit as st
import pandas as pd

# Carica il modello salvato accanto a questo file
BASE_DIR = os.path.dirname(__file__)
pkl_path = os.path.join(BASE_DIR, "ridge_pipeline.pkl")
model = joblib.load(pkl_path)

# Titolo dell'app
st.title("Airbnb Italy Price Prediction")

# Sidebar per input utente
st.sidebar.header("Input dei dati")

# Opzione: upload di un file CSV con i dati
uploaded_file = st.sidebar.file_uploader("Carica un file CSV con le feature", type=["csv"])

if uploaded_file:
    # Lettura del CSV e predizione
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df["predicted_price"] = preds
    st.subheader("Risultati predetti")
    st.write(df)
    st.download_button(
        label="Scarica i risultati",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="predictions.csv",
        mime="text/csv"
    )
else:
    # Inserimento manuale dei valori
    st.sidebar.subheader("Inserimento manuale")
    # Esempio di campi: sostituisci con le feature effettive del tuo modello
    room_type = st.sidebar.selectbox("Tipo di stanza", ["Entire home/apt", "Private room", "Shared room"])
    number_of_reviews = st.sidebar.number_input("Numero di recensioni", min_value=0, step=1)
    availability_365 = st.sidebar.number_input("Disponibilità annuale", min_value=0, max_value=365, step=1)

    if st.sidebar.button("Predici prezzo"):
        # Mappatura valori categorici (adatta alla tua pipeline)
        df_input = pd.DataFrame([{  # struttura conforme alle feature della tua pipeline
            "room_type": room_type,
            "number_of_reviews": number_of_reviews,
            "availability_365": availability_365
        }])
        pred = model.predict(df_input)[0]
        st.subheader("Prezzo predetto")
        st.write(f"€ {pred:.2f}")