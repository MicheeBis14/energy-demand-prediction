import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Chargement des données
train = pd.read_csv('data_train.csv', index_col=0)
val = pd.read_csv('data_val.csv', index_col=0)
test = pd.read_csv('data_test.csv', index_col=0)

# Cible
target = 'Total_Demand_KW'

# Features pour la régression linéaire (on enlève la colonne cible)
features = [col for col in train.columns if col != target]

# =========================
# 1. Régression linéaire
# =========================

lr = LinearRegression()
lr.fit(train[features], train[target])

# Prédictions
pred_val_lr = lr.predict(val[features])
pred_test_lr = lr.predict(test[features])

# Évaluation
rmse_val_lr = np.sqrt(mean_squared_error(val[target], pred_val_lr))
mae_val_lr = mean_absolute_error(val[target], pred_val_lr)
rmse_test_lr = np.sqrt(mean_squared_error(test[target], pred_test_lr))
mae_test_lr = mean_absolute_error(test[target], pred_test_lr)

print('Régression linéaire:')
print(f'Validation - RMSE: {rmse_val_lr:.2f}, MAE: {mae_val_lr:.2f}')
print(f'Test - RMSE: {rmse_test_lr:.2f}, MAE: {mae_test_lr:.2f}')

# =========================
# 2. ARIMA (univarié)
# =========================

# On utilise uniquement la série cible sur le train+val pour l'ARIMA
series = pd.concat([train[target], val[target]])

# Recherche d'ordre automatique (ARIMA simple)
# Pour aller plus vite, on fixe p=1, d=1, q=1 (à optimiser plus tard)
model_arima = sm.tsa.ARIMA(series, order=(1,1,1))
model_arima_fit = model_arima.fit()

# Prédiction sur la période de test
start = len(series)
end = start + len(test) - 1
pred_test_arima = model_arima_fit.predict(start=start, end=end, typ='levels')

# Évaluation
rmse_test_arima = np.sqrt(mean_squared_error(test[target], pred_test_arima))
mae_test_arima = mean_absolute_error(test[target], pred_test_arima)

print('\nARIMA:')
print(f'Test - RMSE: {rmse_test_arima:.2f}, MAE: {mae_test_arima:.2f}')

# =========================
# Visualisation
# =========================
plt.figure(figsize=(12,5))
plt.plot(test[target].values, label='Réel')
plt.plot(pred_test_lr, label='Régression linéaire')
plt.plot(pred_test_arima.values, label='ARIMA')
plt.legend()
plt.title('Prédictions sur le set de test (Baseline)')
plt.xlabel('Index')
plt.ylabel('Total_Demand_KW')
plt.tight_layout()
plt.savefig('baseline_predictions.png')
plt.show() 