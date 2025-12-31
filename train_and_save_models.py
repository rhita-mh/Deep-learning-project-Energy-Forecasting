"""
Script pour entraîner et sauvegarder les modèles exactement comme dans le notebook
Ce script exécute le code exact du notebook et sauvegarde les modèles pour l'interface Streamlit
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import math
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("ENTRAÎNEMENT ET SAUVEGARDE DES MODÈLES")
print("=" * 60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n1. Chargement des données...")
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)
df = df.drop_duplicates()
print(f"✓ Données chargées: {len(df)} lignes")

# ============================================================================
# 2. PRÉPARATION DES DONNÉES (identique au notebook Cell 47)
# ============================================================================
print("\n2. Préparation des données...")
target_col = 'Consumption'
feature_cols = [c for c in df.columns if c != target_col]

window_size = 24
train_ratio = 0.8

values = df[[target_col] + feature_cols].values
n_samples = len(values)
train_size = int(n_samples * train_ratio)

train_values = values[:train_size]
test_values = values[train_size - window_size:]

print(f"Total samples: {n_samples:,}")
print(f"Train size: {train_size:,}")
print(f"Test size (with overlap): {len(test_values):,}")

# Scale all columns together
scaler_all = MinMaxScaler()
train_scaled = scaler_all.fit_transform(train_values)
test_scaled = scaler_all.transform(test_values)

n_features = train_scaled.shape[1]
print(f"\nNumber of features (including target): {n_features}")

# Helper function to invert scaling for target
def invert_scale_target(scaled_target: np.ndarray) -> np.ndarray:
    """Invert MinMax scaling for the target column (assumed at index 0)."""
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    inv = scaler_all.inverse_transform(dummy)
    return inv[:, 0]

# Helper function to create sequences
def create_sequences(data: np.ndarray, window: int, univariate: bool = False):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(window, len(data)):
        if univariate:
            X.append(data[i - window:i, 0])
        else:
            X.append(data[i - window:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create sequences
X_train_uni, y_train_uni = create_sequences(train_scaled, window_size, univariate=True)
X_test_uni, y_test_uni = create_sequences(test_scaled, window_size, univariate=True)
X_train_multi, y_train_multi = create_sequences(train_scaled, window_size, univariate=False)
X_test_multi, y_test_multi = create_sequences(test_scaled, window_size, univariate=False)

# Reshape for models that need 3D input
X_train_uni_3d = X_train_uni.reshape((X_train_uni.shape[0], X_train_uni.shape[1], 1))
X_test_uni_3d = X_test_uni.reshape((X_test_uni.shape[0], X_test_uni.shape[1], 1))
X_train_multi_3d = X_train_multi
X_test_multi_3d = X_test_multi

print(f"\n✓ Séquences créées")

# ============================================================================
# 3. ÉVALUATION FUNCTION (identique au notebook Cell 48)
# ============================================================================
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Evaluate model performance and return metrics."""
    y_true_raw = invert_scale_target(y_true)
    y_pred_raw = invert_scale_target(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)
    
    print(f"\n{model_name} Performance:")
    print("=" * 60)
    print(f"RMSE: {rmse:.3f} MW")
    print(f"MAE:  {mae:.3f} MW")
    print(f"R²:   {r2:.4f}")
    print("=" * 60)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "y_true": y_true_raw,
        "y_pred": y_pred_raw
    }

# ============================================================================
# 4. DECISION TREE
# ============================================================================
print("\n3. Entraînement Decision Tree...")
tree = DecisionTreeRegressor(max_depth=10, random_state=42)
tree.fit(X_train_uni, y_train_uni)
y_pred_tree = tree.predict(X_test_uni)
results_tree = evaluate_model(y_test_uni, y_pred_tree, "Decision Tree")

# ============================================================================
# 5. MLP IMPROVED
# ============================================================================
print("\n4. Entraînement MLP (Improved)...")

def lr_schedule(epoch):
    """Learning rate schedule with warmup and cosine decay"""
    initial_lr = 0.001
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        decay_epochs = 45
        return initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / decay_epochs))

mlp_improved = Sequential([
    Dense(256, activation='relu', input_shape=(window_size,), 
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.25),
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.15),
    Dense(1)
])

mlp_improved.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

reduce_lr_improved = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.3, 
    patience=7, 
    min_lr=1e-7,
    verbose=1
)
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
checkpoint = ModelCheckpoint(
    'best_mlp_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("\nTraining Improved MLP Model...")
history_mlp_improved = mlp_improved.fit(
    X_train_uni, y_train_uni,
    validation_split=0.2,
    epochs=70,
    batch_size=64,
    callbacks=[reduce_lr_improved, lr_scheduler, checkpoint],
    verbose=1
)

# Charger les meilleurs poids
if os.path.exists('best_mlp_improved.h5'):
    mlp_improved = load_model('best_mlp_improved.h5', compile=False)
    mlp_improved.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

y_pred_mlp = mlp_improved.predict(X_test_uni, verbose=0).flatten()
results_mlp = evaluate_model(y_test_uni, y_pred_mlp, "MLP (Improved)")

# ============================================================================
# 6. CNN IMPROVED
# ============================================================================
print("\n5. Entraînement CNN (Improved)...")

cnn_improved = Sequential([
    Conv1D(filters=128, kernel_size=3, activation='relu', 
           input_shape=(window_size, 1),
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Conv1D(filters=128, kernel_size=3, activation='relu',
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(filters=64, kernel_size=3, activation='relu',
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=3, activation='relu',
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=3, activation='relu',
           padding='same'),
    BatchNormalization(),
    Conv1D(filters=32, kernel_size=3, activation='relu',
           padding='same'),
    BatchNormalization(),
    Dropout(0.15),
    GlobalAveragePooling1D(),
    Dense(100, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dropout(0.15),
    Dense(1)
])

cnn_improved.compile(
    optimizer=Adam(learning_rate=0.001),  # Augmenté de 0.0008 à 0.001 pour convergence plus rapide
    loss='mse',
    metrics=['mae']
)

# ReduceLROnPlateau plus agressif pour CNN
reduce_lr_cnn = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,  # Réduction plus agressive (0.3 -> 0.2)
    patience=5,  # Réduit plus tôt (7 -> 5)
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_cnn_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("\nTraining Improved CNN Model...")
history_cnn_improved = cnn_improved.fit(
    X_train_uni_3d, y_train_uni,
    validation_split=0.2,
    epochs=90,
    batch_size=64,
    callbacks=[reduce_lr_cnn, checkpoint],
    verbose=1
)

# Charger les meilleurs poids
if os.path.exists('best_cnn_improved.h5'):
    print("📥 Chargement des meilleurs poids CNN...")
    cnn_improved = load_model('best_cnn_improved.h5', compile=False)
    cnn_improved.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print("✓ Meilleurs poids CNN chargés")

y_pred_cnn = cnn_improved.predict(X_test_uni_3d, verbose=0).flatten()
results_cnn = evaluate_model(y_test_uni, y_pred_cnn, "CNN (Improved)")

# ============================================================================
# 7. LSTM IMPROVED UNIVARIATE
# ============================================================================
print("\n6. Entraînement LSTM (Improved Univariate)...")

lstm_improved = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, 
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                  input_shape=(window_size, 1)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
    BatchNormalization(),
    Dropout(0.25),
    LSTM(32, return_sequences=False,
         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

lstm_improved.compile(
    optimizer=Adam(learning_rate=0.001),  # Augmenté de 0.0008 à 0.001
    loss='mse',
    metrics=['mae']
)

# ReduceLROnPlateau plus agressif pour LSTM Univariate
reduce_lr_lstm = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,  # Réduction plus agressive
    patience=5,  # Réduit plus tôt
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_lstm_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("\nTraining Improved LSTM Model...")
history_lstm_improved = lstm_improved.fit(
    X_train_uni_3d, y_train_uni,
    validation_split=0.2,
    epochs=60,
    batch_size=64,
    callbacks=[reduce_lr_lstm, checkpoint],
    verbose=1
)

# Charger les meilleurs poids
if os.path.exists('best_lstm_improved.h5'):
    print("📥 Chargement des meilleurs poids LSTM...")
    lstm_improved = load_model('best_lstm_improved.h5', compile=False)
    lstm_improved.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print("✓ Meilleurs poids LSTM chargés")

y_pred_lstm = lstm_improved.predict(X_test_uni_3d, verbose=0).flatten()
results_lstm = evaluate_model(y_test_uni, y_pred_lstm, "LSTM (Improved)")

# ============================================================================
# 8. LSTM IMPROVED MULTIVARIATE
# ============================================================================
print("\n7. Entraînement LSTM (Improved Multivariate)...")

lstm_multi_improved = Sequential([
    Bidirectional(LSTM(128, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                  input_shape=(window_size, n_features)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
    BatchNormalization(),
    Dropout(0.25),
    LSTM(32, return_sequences=False,
         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

lstm_multi_improved.compile(
    optimizer=Adam(learning_rate=0.001),  # Augmenté de 0.0008 à 0.001
    loss='mse',
    metrics=['mae']
)

# ReduceLROnPlateau plus agressif pour LSTM Multivariate
reduce_lr_lstm_multi = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,  # Réduction plus agressive
    patience=5,  # Réduit plus tôt
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_lstm_multi_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("\nTraining Improved Multivariate LSTM Model...")
history_lstm_multi_improved = lstm_multi_improved.fit(
    X_train_multi_3d, y_train_multi,
    validation_split=0.2,
    epochs=80,
    batch_size=64,
    callbacks=[reduce_lr_lstm_multi, checkpoint],
    verbose=1
)

# Charger les meilleurs poids
if os.path.exists('best_lstm_multi_improved.h5'):
    print("📥 Chargement des meilleurs poids LSTM Multivariate...")
    lstm_multi_improved = load_model('best_lstm_multi_improved.h5', compile=False)
    lstm_multi_improved.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print("✓ Meilleurs poids LSTM Multivariate chargés")

y_pred_lstm_multi = lstm_multi_improved.predict(X_test_multi_3d, verbose=0).flatten()
results_lstm_multi = evaluate_model(y_test_multi, y_pred_lstm_multi, "Multivariate LSTM (Improved)")

# ============================================================================
# 9. SAUVEGARDE DES MODÈLES ET MÉTRIQUES
# ============================================================================
print("\n8. Sauvegarde des modèles...")

os.makedirs('models', exist_ok=True)

# Sauvegarder le scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_all, f)
print("✓ Scaler sauvegardé")

# Sauvegarder les paramètres
params = {
    'window_size': window_size,
    'n_features': n_features,
    'feature_cols': feature_cols,
    'target_col': target_col
}
with open('models/params.pkl', 'wb') as f:
    pickle.dump(params, f)
print("✓ Paramètres sauvegardés")

# Sauvegarder Decision Tree
with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(tree, f)
print("✓ Decision Tree sauvegardé")

# Sauvegarder les modèles Keras
mlp_improved.save('models/mlp_model.h5')
print("✓ MLP sauvegardé")

cnn_improved.save('models/cnn_model.h5')
print("✓ CNN sauvegardé")

lstm_improved.save('models/lstm_uni_model.h5')
print("✓ LSTM (Univariate) sauvegardé")

lstm_multi_improved.save('models/lstm_multi_model.h5')
print("✓ LSTM (Multivariate) sauvegardé")

# Sauvegarder les métriques
model_metrics = {
    'Decision Tree': {
        'RMSE': results_tree['RMSE'],
        'MAE': results_tree['MAE'],
        'R2': results_tree['R2']
    },
    'MLP': {
        'RMSE': results_mlp['RMSE'],
        'MAE': results_mlp['MAE'],
        'R2': results_mlp['R2']
    },
    'CNN': {
        'RMSE': results_cnn['RMSE'],
        'MAE': results_cnn['MAE'],
        'R2': results_cnn['R2']
    },
    'LSTM (Univariate)': {
        'RMSE': results_lstm['RMSE'],
        'MAE': results_lstm['MAE'],
        'R2': results_lstm['R2']
    },
    'LSTM (Multivariate)': {
        'RMSE': results_lstm_multi['RMSE'],
        'MAE': results_lstm_multi['MAE'],
        'R2': results_lstm_multi['R2']
    },
    'Persistent (Naïve)': {
        'RMSE': 312.370,
        'MAE': 237.482,
        'R2': 0.904
    },
    'ARIMA(1, 0, 0)': {
        'RMSE': 1058.763,
        'MAE': 889.842,
        'R2': -0.100
    }
}

with open('models/model_metrics.pkl', 'wb') as f:
    pickle.dump(model_metrics, f)
print("✓ Métriques sauvegardées")

print("\n" + "=" * 60)
print("✓ TOUS LES MODÈLES ONT ÉTÉ ENTRÂINÉS ET SAUVEGARDÉS")
print("✓ MÉTRIQUES CALCULÉES ET SAUVEGARDÉES")
print("=" * 60)

print("\nMétriques finales:")
print("=" * 60)
for model_name, metrics in model_metrics.items():
    print(f"{model_name}:")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  R²:   {metrics['R2']:.4f}")
print("=" * 60)

print("\n✅ Les modèles sont prêts à être utilisés dans l'interface Streamlit !")
print("   Exécutez: streamlit run app.py")

