import os, joblib
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    pkl_path = os.path.join(BASE_DIR, "ridge_pipeline.pkl")
    if not os.path.exists(pkl_path):
        # In CI o ambienti senza modello, torna None
        return None
    return joblib.load(pkl_path)

def main():
    model = load_model()
    st.title("Airbnb Italy Price Prediction")
    # ... resto della UI ...
    # usa `model` solo se non è None

if __name__ == "__main__":
    main()
