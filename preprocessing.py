import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Chargement des données
DATA_PATH = 'city-hall-electricity-use.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['DateTime_Measured'])

# 2. Nettoyage des données
# Suppression des valeurs nulles ou aberrantes (Total_Demand_KW = 0)
df = df[df['Total_Demand_KW'] > 0].copy()

# Suppression des doublons de timestamps (on garde la moyenne)
df = df.groupby('DateTime_Measured', as_index=False)['Total_Demand_KW'].mean()

# Vérification de la continuité temporelle
# On crée un index temporel complet et on détecte les trous
df = df.sort_values('DateTime_Measured')
df = df.set_index('DateTime_Measured')
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15T')
df = df.reindex(full_range)

# On garde la colonne d'origine pour la demande
# Les valeurs manquantes sont interpolées linéairement
missing_before = df['Total_Demand_KW'].isna().sum()
df['Total_Demand_KW'] = df['Total_Demand_KW'].interpolate(method='linear')
missing_after = df['Total_Demand_KW'].isna().sum()

# 3. Feature engineering
# Ajout de variables temporelles

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['quarter'] = df.index.quarter
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Ajout de lags (décalages temporels)
for lag in [1, 4, 96]:  # 15 min, 1h, 1 jour
    df[f'lag_{lag}'] = df['Total_Demand_KW'].shift(lag)

# Suppression des premières lignes avec NaN dues aux lags
df = df.dropna()

# 4. Normalisation/Standardisation
features = [col for col in df.columns if col != 'Total_Demand_KW']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 5. Split train/val/test (70/15/15)
N = len(df)
train_end = int(0.7 * N)
val_end = int(0.85 * N)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

# Sauvegarde des datasets

df_train.to_csv('data_train.csv')
df_val.to_csv('data_val.csv')
df_test.to_csv('data_test.csv')

print(f"Prétraitement terminé. Données sauvegardées : data_train.csv, data_val.csv, data_test.csv")
print(f"Valeurs manquantes avant interpolation : {missing_before}, après : {missing_after}") 