import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Paramètres du modèle
SEQ_LEN = 24  # 6h d'historique (24 x 15min)
BATCH_SIZE = 64
EPOCHS = 20

# Chargement des données
train = pd.read_csv('data_train.csv', index_col=0)
val = pd.read_csv('data_val.csv', index_col=0)
test = pd.read_csv('data_test.csv', index_col=0)

target = 'Total_Demand_KW'
features = [col for col in train.columns if col != target]

# Fonction pour créer les séquences pour LSTM
def create_sequences(df, seq_len, features, target):
    X, y = [], []
    data = df[features + [target]].values
    for i in range(seq_len, len(df)):
        X.append(data[i-seq_len:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

# Préparation des données
X_train, y_train = create_sequences(train, SEQ_LEN, features, target)
X_val, y_val = create_sequences(val, SEQ_LEN, features, target)
X_test, y_test = create_sequences(test, SEQ_LEN, features, target)

# Modèle LSTM
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, len(features)), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)

# Prédiction sur le test
pred_test = model.predict(X_test).flatten()

# Évaluation
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test = mean_absolute_error(y_test, pred_test)
print(f'LSTM - Test RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}')

# Visualisation
plt.figure(figsize=(12,5))
plt.plot(y_test, label='Réel')
plt.plot(pred_test, label='LSTM')
plt.legend()
plt.title('Prédictions sur le set de test (LSTM)')
plt.xlabel('Index')
plt.ylabel('Total_Demand_KW')
plt.tight_layout()
plt.savefig('lstm_predictions.png')
plt.show() 