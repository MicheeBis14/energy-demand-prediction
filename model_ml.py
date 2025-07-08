import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# XGBoost et LightGBM
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# Chargement des données
train = pd.read_csv('data_train.csv', index_col=0)
val = pd.read_csv('data_val.csv', index_col=0)
test = pd.read_csv('data_test.csv', index_col=0)

target = 'Total_Demand_KW'
features = [col for col in train.columns if col != target]

results = {}

# =========================
# 1. RandomForest
# =========================
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train[features], train[target])
pred_test_rf = rf.predict(test[features])
rmse_rf = np.sqrt(mean_squared_error(test[target], pred_test_rf))
mae_rf = mean_absolute_error(test[target], pred_test_rf)
results['RandomForest'] = (rmse_rf, mae_rf)
print(f'RandomForest - Test RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}')

# =========================
# 2. XGBoost
# =========================
if XGBRegressor is not None:
    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb.fit(train[features], train[target])
    pred_test_xgb = xgb.predict(test[features])
    rmse_xgb = np.sqrt(mean_squared_error(test[target], pred_test_xgb))
    mae_xgb = mean_absolute_error(test[target], pred_test_xgb)
    results['XGBoost'] = (rmse_xgb, mae_xgb)
    print(f'XGBoost - Test RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}')
else:
    print("XGBoost n'est pas installé.")

# =========================
# 3. LightGBM
# =========================
if LGBMRegressor is not None:
    lgbm = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    lgbm.fit(train[features], train[target])
    pred_test_lgbm = lgbm.predict(test[features])
    rmse_lgbm = np.sqrt(mean_squared_error(test[target], pred_test_lgbm))
    mae_lgbm = mean_absolute_error(test[target], pred_test_lgbm)
    results['LightGBM'] = (rmse_lgbm, mae_lgbm)
    print(f'LightGBM - Test RMSE: {rmse_lgbm:.2f}, MAE: {mae_lgbm:.2f}')
else:
    print("LightGBM n'est pas installé.")

# =========================
# Visualisation
# =========================
plt.figure(figsize=(12,5))
plt.plot(test[target].values, label='Réel', color='black')
plt.plot(pred_test_rf, label='RandomForest')
if XGBRegressor is not None:
    plt.plot(pred_test_xgb, label='XGBoost')
if LGBMRegressor is not None:
    plt.plot(pred_test_lgbm, label='LightGBM')
plt.legend()
plt.title('Prédictions sur le set de test (ML)')
plt.xlabel('Index')
plt.ylabel('Total_Demand_KW')
plt.tight_layout()
plt.savefig('ml_predictions.png')
plt.show()

# =========================
# Résumé des scores
# =========================
print("\nRésumé des scores (Test) :")
for model, (rmse, mae) in results.items():
    print(f"{model}: RMSE={rmse:.2f}, MAE={mae:.2f}") 