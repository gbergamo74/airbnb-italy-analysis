import os
import io
import joblib
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
#  UTILIDADES
# ============================================================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ridge_pipeline.pkl")

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """
    Carga el modelo serializado (.pkl). Devuelve None si no existe.
    Se cachea para no recargarlo en cada interacción.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def df_desde_payload(payload: dict, columnas_modelo):
    """
    Crea un DataFrame con TODAS las columnas que el modelo espera.
    Las que falten en el payload se ponen como NaN (el Imputer las manejará).
    """
    fila = {col: payload.get(col, None) for col in columnas_modelo}
    return pd.DataFrame([fila])

def predecir_df(df: pd.DataFrame, modelo):
    """
    Re-ordena/añade columnas faltantes para que coincidan con el modelo y calcula predicciones.
    """
    columnas = list(modelo.feature_names_in_) if hasattr(modelo, "feature_names_in_") else df.columns.tolist()
    df_alineado = df.reindex(columns=columnas)  # columnas nuevas -> NaN
    preds = modelo.predict(df_alineado)
    return preds


# ============================================================
#  APP STREAMLIT
# ============================================================

def main():
    st.title("Airbnb Italy Price Prediction")

    modelo = cargar_modelo()
    if modelo is None:
        st.error("No se encontró el modelo (`ridge_pipeline.pkl`). Súbelo al repositorio/app.")
        st.stop()

    # Columnas que el modelo espera (si se guardaron durante el fit)
    columnas_modelo = list(modelo.feature_names_in_) if hasattr(modelo, "feature_names_in_") else []

    # ------------------- SIDEBAR -------------------
    st.sidebar.header("Entrada de datos")

    # ---- DESCARGAR PLANTILLA CSV ----
    if columnas_modelo:
        plantilla = pd.DataFrame({c: [""] for c in columnas_modelo}).head(5)
        csv_bytes = plantilla.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "Descargar plantilla CSV (5 filas)",
            data=csv_bytes,
            file_name="plantilla_airbnb.csv",
            mime="text/csv"
        )

    # ---- SUBIR CSV ----
    archivo = st.sidebar.file_uploader("Sube un CSV con las variables de entrada", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
        st.subheader("Datos cargados")
        st.write(df.head())

        predicciones = predecir_df(df, modelo)
        df["predicted_price"] = predicciones
        st.subheader("Resultados de la predicción")
        st.write(df)

        # Descargar resultados
        st.download_button(
            "Descargar resultados (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predicciones.csv",
            mime="text/csv"
        )

        # Gráfico de distribución de precios
        st.subheader("Distribución de los precios predichos")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("predicted_price:Q", bin=alt.Bin(maxbins=30), title="Precio predicho"),
            y=alt.Y("count()", title="Conteo")
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

    # ---- ENTRADA MANUAL ----
    st.sidebar.subheader("Entrada manual de valores (simulación rápida)")
    # Ejemplo simple con 3 variables (ajusta según tu modelo real)
    room_type = st.sidebar.selectbox("Tipo de habitación", ["Entire home/apt", "Private room", "Shared room"])
    number_of_reviews = st.sidebar.number_input("Número de reseñas", 0, 10000, 0)
    availability_365 = st.sidebar.number_input("Disponibilidad anual", 0, 365, 0)

    # DataFrame en sesión para acumular simulaciones
    if "simulaciones" not in st.session_state:
        st.session_state.simulaciones = pd.DataFrame(columns=columnas_modelo + ["predicted_price"])

    if st.sidebar.button("Predecir precio"):
        payload = {
            "room_type": room_type,
            "number_of_reviews": number_of_reviews,
            "availability_365": availability_365
        }

        fila_df = df_desde_payload(payload, columnas_modelo)
        pred = float(modelo.predict(fila_df)[0])
        fila_df["predicted_price"] = pred

        st.session_state.simulaciones = pd.concat(
            [st.session_state.simulaciones, fila_df],
            ignore_index=True
        )
        st.success(f"Precio predicho: € {pred:.2f}")

    # Mostrar simulaciones acumuladas
    if not st.session_state.simulaciones.empty:
        st.subheader("Simulaciones manuales acumuladas")
        st.dataframe(st.session_state.simulaciones)

        st.subheader("Evolución de las predicciones manuales")
        sim_df_plot = st.session_state.simulaciones.reset_index().rename(columns={"index": "simulation_id"})
        line = alt.Chart(sim_df_plot).mark_line(point=True).encode(
            x="simulation_id:O",
            y="predicted_price:Q",
            tooltip=list(sim_df_plot.columns)
        )
        st.altair_chart(line, use_container_width=True)

        if st.button("Reiniciar simulaciones"):
            st.session_state.simulaciones = pd.DataFrame(columns=columnas_modelo + ["predicted_price"])
            st.rerun()  # reinicia la app para limpiar


if __name__ == "__main__":
    main()
