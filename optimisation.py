import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Chargement des données
train = pd.read_csv('data_train.csv', index_col=0)
val = pd.read_csv('data_val.csv', index_col=0)
test = pd.read_csv('data_test.csv', index_col=0)
features = [col for col in train.columns if col != 'Total_Demand_KW']
target = 'Total_Demand_KW'

# On combine train et val pour le tuning
X_trainval = pd.concat([train[features], val[features]])
y_trainval = pd.concat([train[target], val[target]])

# Grille d'hyperparamètres
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63]
}

lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
gs = GridSearchCV(lgbm, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=2)
gs.fit(X_trainval, y_trainval)

print(f"Meilleurs hyperparamètres : {gs.best_params_}")

# Évaluation sur le test
best_lgbm = gs.best_estimator_
pred_test = best_lgbm.predict(test[features])
rmse = np.sqrt(mean_squared_error(test[target], pred_test))
mae = mean_absolute_error(test[target], pred_test)
print(f"LightGBM optimisé - Test RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Comparaison avec le modèle par défaut
lgbm_default = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lgbm_default.fit(X_trainval, y_trainval)
pred_test_default = lgbm_default.predict(test[features])
rmse_default = np.sqrt(mean_squared_error(test[target], pred_test_default))
mae_default = mean_absolute_error(test[target], pred_test_default)
print(f"LightGBM défaut - Test RMSE: {rmse_default:.2f}, MAE: {mae_default:.2f}")

joblib.dump(best_lgbm, "lgbm_optimise.pkl")
print("Modèle LightGBM optimisé sauvegardé sous lgbm_optimise.pkl") 