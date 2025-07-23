# ============================================================
#  Airbnb Italy Price Prediction - Streamlit App
#  Comentarios en español para los profesores.
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
#  RUTAS Y UTILIDADES
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ridge_pipeline.pkl")

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """
    Carga el modelo serializado (.pkl). Devuelve None si no está presente.
    Se cachea para que no se recargue en cada interacción.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def alinear_columnas(df, modelo):
    """
    Reordena y agrega columnas faltantes para que coincidan con las usadas al entrenar el modelo.
    Las columnas faltantes se rellenan con NaN (el Imputer del pipeline se encarga).
    """
    if hasattr(modelo, "feature_names_in_"):
        columnas = list(modelo.feature_names_in_)
        df_alineado = df.reindex(columns=columnas)
    else:
        # Si el modelo no guarda las columnas, usamos lo que venga.
        df_alineado = df
    return df_alineado

def crear_payload_df(payload_dic, columnas_modelo):
    """
    Crea un DataFrame de una sola fila con TODAS las columnas del modelo.
    """
    fila = {c: payload_dic.get(c, None) for c in columnas_modelo}
    return pd.DataFrame([fila])

# ------------------------------------------------------------
#  APLICACIÓN STREAMLIT
# ------------------------------------------------------------

def main():
    st.title("Airbnb Italy Price Prediction")

    # 1) Cargar modelo
    modelo = cargar_modelo()
    if modelo is None:
        st.error("No se encontró el modelo (`ridge_pipeline.pkl`). Súbelo al repositorio / carpeta de la app.")
        st.stop()

    # Columnas esperadas por el modelo (si se guardaron)
    columnas_modelo = list(getattr(modelo, "feature_names_in_", []))

    # --------------------- SIDEBAR ----------------------------
    st.sidebar.header("Entrada de datos")

    # 1.1) Botón para descargar plantilla CSV (5 filas vacías)
    if columnas_modelo:
        plantilla = pd.DataFrame({c: [""] for c in columnas_modelo}).head(5)
        st.sidebar.download_button(
            label="Descargar plantilla CSV (5 filas)",
            data=plantilla.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_airbnb.csv",
            mime="text/csv"
        )

    # 1.2) Subida de CSV
    archivo = st.sidebar.file_uploader("Sube un CSV con las variables de entrada", type=["csv"])
    df_pred = None    # DataFrame que almacenará los resultados, si hay CSV
    if archivo is not None:
        df_input = pd.read_csv(archivo)
        st.subheader("Datos cargados")
        st.write(df_input.head())

        # Alinear columnas y predecir
        df_alineado = alinear_columnas(df_input, modelo)
        predicciones = modelo.predict(df_alineado)
        df_input["predicted_price"] = predicciones

        # Evitar precios negativos
        df_input["predicted_price"] = df_input["predicted_price"].clip(lower=0)

        st.subheader("Resultados de la predicción")
        st.write(df_input.head(20))   # mostramos primeras 20 filas
        df_pred = df_input.copy()

        # Botón para descargar resultados
        st.download_button(
            "Descargar resultados (CSV)",
            data=df_input.to_csv(index=False).encode("utf-8"),
            file_name="predicciones.csv",
            mime="text/csv"
        )

        st.markdown("---")

    # ------------------ ENTRADA MANUAL ------------------------
    st.sidebar.subheader("Entrada manual de valores (simulación rápida)")

    # NOTA: Ajusta estas variables para que coincidan con tu modelo real.
    # Aquí usamos tres que sabemos que existen en el dataset original.
    room_type = st.sidebar.selectbox("Tipo de habitación", ["Entire home/apt", "Private room", "Shared room"])
    number_of_reviews = st.sidebar.number_input("Número de reseñas", 0, 10000, 0)
    availability_365 = st.sidebar.number_input("Disponibilidad anual", 0, 365, 0)

    # Creamos un DataFrame en sesión para acumular simulaciones
    if "simulaciones" not in st.session_state:
        cols_sim = columnas_modelo + ["predicted_price"] if columnas_modelo else ["room_type","number_of_reviews","availability_365","predicted_price"]
        st.session_state.simulaciones = pd.DataFrame(columns=cols_sim)

    if st.sidebar.button("Predecir precio"):
        payload = {
            "room_type": room_type,
            "number_of_reviews": number_of_reviews,
            "availability_365": availability_365
        }
        if columnas_modelo:
            fila_df = crear_payload_df(payload, columnas_modelo)
        else:
            fila_df = pd.DataFrame([payload])
        pred = float(modelo.predict(alinear_columnas(fila_df, modelo))[0])
        fila_df["predicted_price"] = max(pred, 0)  # clip a 0
        st.session_state.simulaciones = pd.concat([st.session_state.simulaciones, fila_df], ignore_index=True)
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
            st.session_state.simulaciones = st.session_state.simulaciones.iloc[0:0]
            st.rerun()

    # ------------------ ANALÍTICA EXTRA (si hay df_pred) ------------------
    if df_pred is not None and "predicted_price" in df_pred.columns:
        st.markdown("---")
        st.header("Análisis adicional de las predicciones")

        # Estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        stats = df_pred["predicted_price"].describe().rename({
            "count": "conteo", "mean": "media", "std": "desv_std",
            "min": "mínimo", "25%": "q25", "50%": "mediana", "75%": "q75", "max": "máximo"
        })
        st.write(stats)

        # Histograma (Altair)
        st.subheader("Distribución de los precios predichos")
        # Para gráficos más rápidos con datasets grandes, muestreamos
        df_plot = df_pred.sample(5000, random_state=0) if len(df_pred) > 5000 else df_pred
        hist = alt.Chart(df_plot).mark_bar().encode(
            x=alt.X("predicted_price:Q", bin=alt.Bin(maxbins=50), title="Precio predicho"),
            y=alt.Y("count()", title="Conteo")
        )
        st.altair_chart(hist, use_container_width=True)

        # Boxplot por categoría (si existe una columna categórica principal)
        st.subheader("Distribución por categoría")
        cat_cols = [c for c in df_pred.columns if df_pred[c].dtype == "object" and c not in ["listing_url"]]  # heurística
        if cat_cols:
            cat_col = st.selectbox("Elige columna categórica", cat_cols, index=0)
            box = alt.Chart(df_plot).mark_boxplot().encode(
                x=alt.X(f"{cat_col}:N", title=cat_col),
                y=alt.Y("predicted_price:Q", title="Precio predicho")
            )
            st.altair_chart(box, use_container_width=True)
        else:
            st.info("No se detectaron columnas categóricas para el boxplot.")

        # Scatter interactivo
        st.subheader("Relación entre el precio predicho y otra variable numérica")
        num_cols = [c for c in df_pred.columns if pd.api.types.is_numeric_dtype(df_pred[c]) and c != "predicted_price"]
        if num_cols:
            x_sel = st.selectbox("Variable numérica (X):", num_cols, index=0)
            scatter = alt.Chart(df_plot).mark_circle(size=40, opacity=0.4).encode(
                x=alt.X(f"{x_sel}:Q", title=x_sel),
                y=alt.Y("predicted_price:Q", title="Precio predicho"),
                tooltip=list(df_pred.columns)
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("No hay columnas numéricas (aparte de la predicción) para el scatter.")

        # Top / Bottom N
        st.subheader("Top / Bottom anuncios por precio predicho")
        N = st.slider("¿Cuántos mostrar?", 5, 50, 10)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Top {N}")
            st.dataframe(df_pred.nlargest(N, "predicted_price"))
        with col2:
            st.write(f"Bottom {N}")
            st.dataframe(df_pred.nsmallest(N, "predicted_price"))

        # Evaluación rápida si el CSV trae el precio real
        if "price" in df_pred.columns:
            st.subheader("Evaluación rápida (si hay 'price' real en el CSV)")
            df_pred["residuo"] = df_pred["price"] - df_pred["predicted_price"]
            mae = df_pred["residuo"].abs().mean()
            rmse = (df_pred["residuo"]**2).mean() ** 0.5
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")

            resid_hist = alt.Chart(df_pred).mark_bar().encode(
                x=alt.X("residuo:Q", bin=alt.Bin(maxbins=40), title="Residuo (real - pred)"),
                y="count()"
            )
            st.altair_chart(resid_hist, use_container_width=True)

        # Importancia de características (coeficientes Ridge)
        st.subheader("Importancia aproximada de características (coeficientes Ridge)")
        try:
            ridge = None
            for name, step in getattr(modelo, "named_steps", {}).items():
                if isinstance(step, Ridge):
                    ridge = step
                    break
            if ridge is not None and hasattr(ridge, "coef_"):
                # Intentamos recuperar nombres finales
                try:
                    feat_names = list(modelo.feature_names_in_)
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(ridge.coef_))]

                imp_df = (pd.DataFrame({"feature": feat_names, "coef": ridge.coef_})
                            .assign(abs_coef=lambda d: d["coef"].abs())
                            .sort_values("abs_coef", ascending=False)
                            .head(20))
                bar_imp = alt.Chart(imp_df).mark_bar().encode(
                    x=alt.X("abs_coef:Q", title="|coeficiente|"),
                    y=alt.Y("feature:N", sort='-x', title="feature"),
                    tooltip=["feature", "coef"]
                )
                st.altair_chart(bar_imp, use_container_width=True)
            else:
                st.info("No se encontró un estimador Ridge dentro del pipeline o no tiene coeficientes.")
        except Exception as e:
            st.info(f"No se pudieron mostrar los coeficientes del modelo: {e}")

# ------------------------------------------------------------
#  EJECUCIÓN
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
