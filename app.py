import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction de la demande d'énergie renouvelable", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller à :", [
    "Accueil / Présentation",
    "Visualisation des données",
    "Comparaison des modèles",
    "Importance des variables",
    "Prédiction personnalisée",
    "Téléchargement des résultats",
    "(Bonus) Prophet ou PCA"
])

# 1. Accueil / Présentation
def accueil():
    st.title("Prédiction de la demande d'énergie renouvelable")
    st.markdown("""
    **Projet complet de data science :**
    - Prévision de la consommation électrique à partir de données réelles (Boston City Hall)
    - Machine Learning, Deep Learning, séries temporelles
    - Visualisations, évaluation, optimisation, documentation
    
    _Auteur : Abderrahman AJINOU – Université Paris Cité_
    """)
    st.image("compare_models.png", caption="Comparaison des modèles", use_column_width=True)

# 2. Visualisation des données
def visualisation():
    st.header("Exploration des données")
    df = pd.read_csv("data_train.csv", index_col=0)
    st.write("Aperçu des données d'entraînement :", df.head())
    st.write(f"Nombre de lignes : {len(df)}")
    st.line_chart(df['Total_Demand_KW'].head(1000))
    st.image("zoom_predictions.png", caption="Zoom sur les 500 premières prédictions")
    st.image("error_distribution.png", caption="Distribution des erreurs")

# 3. Comparaison des modèles
def comparaison():
    st.header("Comparaison des modèles")
    st.image("compare_models.png", caption="Comparaison des prédictions sur le test")
    st.markdown("**Scores principaux :**")
    st.markdown("""
    | Modèle                | RMSE   | MAE   | MAPE (%) |
    |-----------------------|--------|-------|----------|
    | Régression linéaire   | 26.54  | 13.82 | 1.29     |
    | RandomForest          | 33.65  | 16.88 | 1.71     |
    | XGBoost               | 138.99 | 69.95 | 8.49     |
    | LightGBM (défaut)     | 34.90  | 19.35 | 2.08     |
    | LightGBM (optimisé)   | 31.61  | 16.68 | -        |
    | LSTM                  | 40.73  | 23.34 | -        |
    """)

# 4. Importance des variables
def importance():
    st.header("Importance des variables (RandomForest)")
    st.image("feature_importance_rf.png", caption="Importance des variables explicatives")

# 5. Prédiction personnalisée
def prediction():
    st.header("Prédiction personnalisée")
    st.write("Entrez les valeurs des features pour obtenir une prédiction (modèle LightGBM optimisé)")
    # Pour simplifier, on propose les features principales
    hour = st.slider("Heure", 0, 23, 12)
    dayofweek = st.slider("Jour de la semaine (0=lundi)", 0, 6, 2)
    month = st.slider("Mois", 1, 12, 6)
    year = st.slider("Année", 2016, 2020, 2019)
    quarter = st.slider("Trimestre", 1, 4, 2)
    is_weekend = st.selectbox("Week-end ?", [0, 1])
    lag_1 = st.number_input("Demande précédente (15min)", value=1400.0)
    lag_4 = st.number_input("Demande précédente (1h)", value=1400.0)
    lag_96 = st.number_input("Demande précédente (1 jour)", value=1400.0)
    # Normalisation approximative (pour la démo)
    X = np.array([[hour, dayofweek, month, year, quarter, is_weekend, lag_1, lag_4, lag_96]])
    # Charger le modèle LightGBM optimisé
    import joblib
    try:
        model = joblib.load("lgbm_optimise.pkl")
        pred = model.predict(X)[0]
        st.success(f"Prédiction de la demande (kW) : {pred:.2f}")
    except Exception as e:
        st.warning("Modèle non disponible. Veuillez entraîner et sauvegarder le modèle LightGBM optimisé sous 'lgbm_optimise.pkl'.")

# 6. Téléchargement des résultats
def telechargement():
    st.header("Téléchargement des résultats et visualisations")
    for img in ["compare_models.png", "zoom_predictions.png", "error_distribution.png", "feature_importance_rf.png"]:
        with open(img, "rb") as f:
            st.download_button(f"Télécharger {img}", f, file_name=img)

# 7. Bonus Prophet ou PCA
def bonus():
    st.header("Bonus : Prophet ou PCA")
    choix = st.radio("Choisir l'analyse avancée :", ["Prophet (prévision)", "PCA (réduction de dimension)"])
    if choix == "Prophet (prévision)":
        st.subheader("Prévision avec Prophet")
        try:
            from prophet import Prophet
        except ImportError:
            st.warning("La librairie Prophet n'est pas installée. Installez-la avec 'pip install prophet'.")
            return
        df = pd.read_csv("data_train.csv", index_col=0)
        # On suppose qu'il y a une colonne 'date' ou on la reconstitue
        if 'date' in df.columns:
            df_prophet = df[['date', 'Total_Demand_KW']].rename(columns={'date': 'ds', 'Total_Demand_KW': 'y'})
        else:
            df_prophet = df.copy()
            df_prophet['ds'] = pd.to_datetime(df_prophet.index)
            df_prophet = df_prophet[['ds', 'Total_Demand_KW']].rename(columns={'Total_Demand_KW': 'y'})
        horizon = st.selectbox("Horizon de prévision", [24, 96, 672], format_func=lambda x: f"{x//96} semaine(s)" if x>=96 else f"{x} pas de temps (15min)")
        if st.button("Lancer la prévision Prophet"):
            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=horizon, freq='15min')
            forecast = m.predict(future)
            st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].tail(horizon))
            st.write("Composantes de la prévision :")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
    elif choix == "PCA (réduction de dimension)":
        st.subheader("Réduction de dimension avec PCA")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        df = pd.read_csv("data_train.csv", index_col=0)
        features = [col for col in df.columns if col != 'Total_Demand_KW']
        X = df[features].select_dtypes(include=[np.number]).dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        st.write(f"Variance expliquée : {pca.explained_variance_ratio_[0]:.2%}, {pca.explained_variance_ratio_[1]:.2%}")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.3)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Projection PCA (2 composantes)')
        st.pyplot(fig)

# Routing
if section == "Accueil / Présentation":
    accueil()
elif section == "Visualisation des données":
    visualisation()
elif section == "Comparaison des modèles":
    comparaison()
elif section == "Importance des variables":
    importance()
elif section == "Prédiction personnalisée":
    prediction()
elif section == "Téléchargement des résultats":
    telechargement()
elif section == "(Bonus) Prophet ou PCA":
    bonus() 