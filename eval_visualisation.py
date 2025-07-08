import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Pour MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Chargement des vraies valeurs du test
true = pd.read_csv('data_test.csv', index_col=0)['Total_Demand_KW'].values
train = pd.read_csv('data_train.csv', index_col=0)
test = pd.read_csv('data_test.csv', index_col=0)
features = [col for col in train.columns if col != 'Total_Demand_KW']

# Modèles déjà entraînés
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

lr = LinearRegression().fit(train[features], train['Total_Demand_KW'])
pred_lr = lr.predict(test[features])
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[features], train['Total_Demand_KW'])
pred_rf = rf.predict(test[features])
xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[features], train['Total_Demand_KW'])
pred_xgb = xgb.predict(test[features])
lgbm = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[features], train['Total_Demand_KW'])
pred_lgbm = lgbm.predict(test[features])

models = {
    'Régression linéaire': pred_lr,
    'RandomForest': pred_rf,
    'XGBoost': pred_xgb,
    'LightGBM': pred_lgbm,
}

# 1. Calcul des métriques
print("\nMétriques sur le set de test :")
print("Modèle\t\tRMSE\t\tMAE\t\tMAPE (%)")
for name, pred in models.items():
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    print(f"{name:18s} {rmse:8.2f} {mae:8.2f} {mape:8.2f}")

# 2. Visualisation détaillée (zoom sur 500 premiers points)
plt.figure(figsize=(14,6))
plt.plot(true[:500], label='Réel', color='black', linewidth=1)
for name, pred in models.items():
    plt.plot(pred[:500], label=name, alpha=0.7)
plt.legend()
plt.title('Zoom sur les 500 premières prédictions')
plt.xlabel('Index')
plt.ylabel('Total_Demand_KW')
plt.tight_layout()
plt.savefig('zoom_predictions.png')
plt.show()

# 3. Distribution des erreurs
plt.figure(figsize=(10,6))
for name, pred in models.items():
    errors = true - pred
    plt.hist(errors, bins=50, alpha=0.5, label=name)
plt.legend()
plt.title('Distribution des erreurs de prédiction')
plt.xlabel('Erreur (kW)')
plt.ylabel('Fréquence')
plt.tight_layout()
plt.savefig('error_distribution.png')
plt.show()

# 4. Importance des variables (RandomForest)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,5))
plt.title("Importance des variables (RandomForest)")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show() 