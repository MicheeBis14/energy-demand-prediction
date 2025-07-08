import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement des vraies valeurs du test
true = pd.read_csv('data_test.csv', index_col=0)['Total_Demand_KW'].values

# Chargement des prédictions
# Baseline
from sklearn.linear_model import LinearRegression
train = pd.read_csv('data_train.csv', index_col=0)
val = pd.read_csv('data_val.csv', index_col=0)
test = pd.read_csv('data_test.csv', index_col=0)
features = [col for col in train.columns if col != 'Total_Demand_KW']
lr = LinearRegression().fit(train[features], train['Total_Demand_KW'])
pred_lr = lr.predict(test[features])

# RandomForest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[features], train['Total_Demand_KW'])
pred_rf = rf.predict(test[features])

# XGBoost
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[features], train['Total_Demand_KW'])
pred_xgb = xgb.predict(test[features])

# LightGBM
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[features], train['Total_Demand_KW'])
pred_lgbm = lgbm.predict(test[features])

# LSTM (chargement des prédictions depuis le fichier généré)
import os
if os.path.exists('lstm_predictions.png'):
    # On suppose que le script LSTM a déjà affiché la courbe, on recharge les valeurs
    # Pour une vraie pipeline, il faudrait sauvegarder les prédictions dans un fichier .npy ou .csv
    # Ici, on relance le modèle pour obtenir les prédictions
    from model_lstm import create_sequences, features, target, SEQ_LEN
    X_test, y_test = create_sequences(test, SEQ_LEN, features, target)
    from tensorflow.keras.models import load_model, Sequential
    # On suppose que le modèle est encore en mémoire, sinon il faudrait le sauvegarder
    # Pour la comparaison, on relance l'entraînement rapide
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, len(features)), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    # On ne réentraîne pas ici pour éviter la lourdeur, donc on ne compare que les autres modèles
    pred_lstm = np.zeros_like(y_test)  # Placeholder
else:
    pred_lstm = np.zeros_like(true)  # Placeholder

from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_scores(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred)

scores = {
    'Régression linéaire': get_scores(true, pred_lr),
    'RandomForest': get_scores(true, pred_rf),
    'XGBoost': get_scores(true, pred_xgb),
    'LightGBM': get_scores(true, pred_lgbm),
    # 'LSTM': get_scores(y_test, pred_lstm),  # À activer si on a les vraies prédictions LSTM
}

# Tableau récapitulatif
print("\nRésumé des scores (Test) :")
print("Modèle\t\tRMSE\t\tMAE")
for model, (rmse, mae) in scores.items():
    print(f"{model:18s} {rmse:8.2f} {mae:8.2f}")

# Visualisation commune
plt.figure(figsize=(14,6))
plt.plot(true, label='Réel', color='black', linewidth=1)
plt.plot(pred_lr, label='Régression linéaire', alpha=0.7)
plt.plot(pred_rf, label='RandomForest', alpha=0.7)
plt.plot(pred_xgb, label='XGBoost', alpha=0.7)
plt.plot(pred_lgbm, label='LightGBM', alpha=0.7)
# plt.plot(pred_lstm, label='LSTM', alpha=0.7)  # À activer si on a les vraies prédictions LSTM
plt.legend()
plt.title('Comparaison des prédictions sur le set de test')
plt.xlabel('Index')
plt.ylabel('Total_Demand_KW')
plt.tight_layout()
plt.savefig('compare_models.png')
plt.show() 