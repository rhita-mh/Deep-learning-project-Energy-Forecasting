"""
Script pour entraîner les modèles en utilisant EXACTEMENT le même code que le notebook
Ce script copie directement le code des cellules du notebook
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("ENTRAÎNEMENT DES MODÈLES (CODE EXACT DU NOTEBOOK)")
print("=" * 60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES (identique au notebook)
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

window_size = 24      # Use 24 past hours to predict the next hour
train_ratio = 0.8     # 80% of data for training

# Prepare data
values = df[[target_col] + feature_cols].values
n_samples = len(values)
train_size = int(n_samples * train_ratio)

train_values = values[:train_size]
test_values = values[train_size - window_size:]  # Include overlap for windows

print(f"Total samples: {n_samples:,}")
print(f"Train size: {train_size:,}")
print(f"Test size (with overlap): {len(test_values):,}")

# Vérifier les valeurs avant normalisation
print(f"\n📊 Vérification avant normalisation:")
print(f"   Train - Consumption min: {train_values[:, 0].min():.2f}, max: {train_values[:, 0].max():.2f}")
print(f"   Test  - Consumption min: {test_values[:, 0].min():.2f}, max: {test_values[:, 0].max():.2f}")

# Scale all columns together
scaler_all = MinMaxScaler()
train_scaled = scaler_all.fit_transform(train_values)
test_scaled = scaler_all.transform(test_values)

# Vérifier les valeurs après normalisation
print(f"\n📊 Vérification après normalisation:")
print(f"   Train - Consumption scaled min: {train_scaled[:, 0].min():.6f}, max: {train_scaled[:, 0].max():.6f}")
print(f"   Test  - Consumption scaled min: {test_scaled[:, 0].min():.6f}, max: {test_scaled[:, 0].max():.6f}")

# Si test_scaled a des valeurs < 0 ou > 1, c'est un problème
if test_scaled[:, 0].min() < -0.01 or test_scaled[:, 0].max() > 1.01:
    print(f"\n   ⚠️ ATTENTION: Les valeurs de test sont hors de [0, 1]!")
    print(f"      Cela signifie que le test set contient des valeurs qui n'étaient pas dans le train set.")
    print(f"      Solution: Utiliser fit sur train+test ou StandardScaler au lieu de MinMaxScaler")
    print(f"      Pour l'instant, on continue mais les résultats peuvent être affectés.")

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
            # Use only target variable (column 0)
            X.append(data[i - window:i, 0])
        else:
            # Use all features
            X.append(data[i - window:i, :])
        y.append(data[i, 0])  # Target is always Consumption (column 0)
    return np.array(X), np.array(y)

# Create sequences for univariate models (MLP, CNN, LSTM)
X_train_uni, y_train_uni = create_sequences(train_scaled, window_size, univariate=True)
X_test_uni, y_test_uni = create_sequences(test_scaled, window_size, univariate=True)

# Create sequences for multivariate models
X_train_multi, y_train_multi = create_sequences(train_scaled, window_size, univariate=False)
X_test_multi, y_test_multi = create_sequences(test_scaled, window_size, univariate=False)

print("\n" + "=" * 60)
print("Sequence Shapes")
print("=" * 60)
print(f"Univariate - X_train: {X_train_uni.shape}, y_train: {y_train_uni.shape}")
print(f"Univariate - X_test: {X_test_uni.shape}, y_test: {y_test_uni.shape}")
print(f"Multivariate - X_train: {X_train_multi.shape}, y_train: {y_train_multi.shape}")
print(f"Multivariate - X_test: {X_test_multi.shape}, y_test: {y_test_multi.shape}")

# Reshape for models that need 3D input (CNN, LSTM)
X_train_uni_3d = X_train_uni.reshape((X_train_uni.shape[0], X_train_uni.shape[1], 1))
X_test_uni_3d = X_test_uni.reshape((X_test_uni.shape[0], X_test_uni.shape[1], 1))
X_train_multi_3d = X_train_multi
X_test_multi_3d = X_test_multi

print(f"\n3D Shapes for CNN/LSTM:")
print(f"Univariate 3D - X_train: {X_train_uni_3d.shape}, X_test: {X_test_uni_3d.shape}")
print(f"Multivariate 3D - X_train: {X_train_multi_3d.shape}, X_test: {X_test_multi_3d.shape}")

# ============================================================================
# 3. ÉVALUATION FUNCTION (identique au notebook Cell 48)
# ============================================================================
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Evaluate model performance and return metrics."""
    # Vérifications préliminaires
    print(f"\n🔍 Diagnostic pour {model_name}:")
    print(f"   y_true (scaled) - Min: {y_true.min():.6f}, Max: {y_true.max():.6f}, Mean: {y_true.mean():.6f}")
    print(f"   y_pred (scaled) - Min: {y_pred.min():.6f}, Max: {y_pred.max():.6f}, Mean: {y_pred.mean():.6f}")
    
    # Vérifier si y_true est dans [0, 1] (devrait l'être pour MinMaxScaler)
    if y_true.min() < -0.01:
        print(f"   ⚠️ ATTENTION: y_true a des valeurs négatives ({y_true.min():.6f})!")
        print(f"      Cela suggère un problème avec la normalisation ou les données de test.")
        print(f"      Vérifiez que le scaler a été entraîné sur les bonnes données.")
    
    # Vérifier si les prédictions sont dans [0, 1]
    if y_pred.min() < -0.1 or y_pred.max() > 1.1:
        print(f"   ⚠️ ATTENTION: Prédictions hors de [0, 1]! Cela peut indiquer un problème.")
    
    # Invert scaling
    y_true_raw = invert_scale_target(y_true)
    y_pred_raw = invert_scale_target(y_pred)
    
    print(f"   y_true_raw - Min: {y_true_raw.min():.2f}, Max: {y_true_raw.max():.2f}, Mean: {y_true_raw.mean():.2f}")
    print(f"   y_pred_raw - Min: {y_pred_raw.min():.2f}, Max: {y_pred_raw.max():.2f}, Mean: {y_pred_raw.mean():.2f}")
    
    # Vérifier si les prédictions sont dans une plage raisonnable
    consumption_min = df['Consumption'].min()
    consumption_max = df['Consumption'].max()
    if y_pred_raw.min() < consumption_min * 0.5 or y_pred_raw.max() > consumption_max * 1.5:
        print(f"   ⚠️ ATTENTION: Prédictions hors de la plage normale [{consumption_min:.2f}, {consumption_max:.2f}]")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)
    
    # Vérifier la corrélation
    correlation = np.corrcoef(y_true_raw, y_pred_raw)[0, 1]
    
    print(f"\n{model_name} Performance:")
    print("=" * 60)
    print(f"RMSE: {rmse:.3f} MW")
    print(f"MAE:  {mae:.3f} MW")
    print(f"R²:   {r2:.4f}")
    print(f"Corrélation: {correlation:.4f}")
    print("=" * 60)
    
    if rmse > 500:
        print(f"   ❌ RMSE très élevé! Causes possibles:")
        print(f"      - Modèle mal entraîné")
        print(f"      - Problème avec les données")
        print(f"      - Problème avec l'inversion de normalisation")
    
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "y_true": y_true_raw,
        "y_pred": y_pred_raw
    }

# ============================================================================
# 4. DECISION TREE (identique au notebook Cell 44)
# ============================================================================
print("\n3. Entraînement Decision Tree...")
X_train_tree = X_train_uni
X_test_tree = X_test_uni

tree = DecisionTreeRegressor(max_depth=10, random_state=42)
tree.fit(X_train_tree, y_train_uni)

y_pred_tree_scaled = tree.predict(X_test_tree)
results_tree = evaluate_model(y_test_uni, y_pred_tree_scaled, "Decision Tree")

# ============================================================================
# 5. MLP IMPROVED (identique au notebook Cell 53)
# ============================================================================
print("\n4. Entraînement MLP (Improved)...")

# Learning rate schedule function
def lr_schedule(epoch):
    """Learning rate schedule with warmup and cosine decay"""
    initial_lr = 0.001
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        decay_epochs = 45
        return initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / decay_epochs))

# Build Improved MLP model with deeper architecture
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

# Callbacks pour MLP - SANS early stopping, nombre d'epochs fixe
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

# Train improved model
print("\nTraining Improved MLP Model...")
history_mlp_improved = mlp_improved.fit(
    X_train_uni, y_train_uni,
    validation_split=0.2,
    epochs=70,  # Nombre exact d'epochs comme dans le notebook
    batch_size=64,  # Larger batch size for stability
    callbacks=[reduce_lr_improved, lr_scheduler, checkpoint],  # SANS early_stop_improved
    verbose=1
)

# Make predictions
y_pred_mlp_improved = mlp_improved.predict(X_test_uni, verbose=0).flatten()

# Evaluate (les métriques seront calculées automatiquement)
results_mlp_improved = evaluate_model(y_test_uni, y_pred_mlp_improved, "MLP (Improved)")
print(f"✓ MLP (Improved) - RMSE: {results_mlp_improved['RMSE']:.3f}, MAE: {results_mlp_improved['MAE']:.3f}, R²: {results_mlp_improved['R2']:.4f}")

# ============================================================================
# 6. CNN IMPROVED (identique au notebook Cell 55)
# ============================================================================
print("\n5. Entraînement CNN (Improved)...")

cnn_improved = Sequential([
    # First block
    Conv1D(filters=128, kernel_size=3, activation='relu', 
           input_shape=(window_size, 1),
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Conv1D(filters=128, kernel_size=3, activation='relu',
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),  # 24 -> 12
    Dropout(0.25),
    
    # Second block
    Conv1D(filters=64, kernel_size=3, activation='relu',
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=3, activation='relu',
           padding='same',
           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),  # 12 -> 6
    Dropout(0.2),
    
    # Third block
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
    optimizer=Adam(learning_rate=0.0008),
    loss='mse',
    metrics=['mae']
)

checkpoint = ModelCheckpoint(
    'best_cnn_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Train improved model
print("\nTraining Improved CNN Model...")
history_cnn_improved = cnn_improved.fit(
    X_train_uni_3d, y_train_uni,
    validation_split=0.2,
    epochs=90,  # Nombre exact d'epochs comme dans le notebook
    batch_size=64,
    callbacks=[reduce_lr_improved, checkpoint],  # SANS early_stop_improved
    verbose=1
)

# IMPORTANT: Charger les meilleurs poids sauvegardés par ModelCheckpoint
# ModelCheckpoint sauvegarde les meilleurs poids, on doit les charger explicitement
try:
    if os.path.exists('best_cnn_improved.h5'):
        print("\n📥 Chargement des meilleurs poids CNN depuis le checkpoint...")
        cnn_improved = load_model('best_cnn_improved.h5', compile=False)
        cnn_improved.compile(optimizer=Adam(learning_rate=0.0008), loss='mse', metrics=['mae'])
        print("✓ Meilleurs poids CNN chargés")
    else:
        print("⚠️ Fichier best_cnn_improved.h5 non trouvé, utilisation du modèle actuel")
except Exception as e:
    print(f"⚠️ Impossible de charger le checkpoint CNN: {e}")
    print("   Utilisation du modèle actuel")

# Make predictions
y_pred_cnn_improved = cnn_improved.predict(X_test_uni_3d, verbose=0).flatten()

# Evaluate (les métriques seront calculées automatiquement)
results_cnn_improved = evaluate_model(y_test_uni, y_pred_cnn_improved, "CNN (Improved)")
print(f"✓ CNN (Improved) - RMSE: {results_cnn_improved['RMSE']:.3f}, MAE: {results_cnn_improved['MAE']:.3f}, R²: {results_cnn_improved['R2']:.4f}")

# ============================================================================
# 7. LSTM IMPROVED UNIVARIATE (identique au notebook Cell 57)
# ============================================================================
print("\n6. Entraînement LSTM (Improved Univariate)...")

lstm_improved = Sequential([
    # First bidirectional LSTM block
    Bidirectional(LSTM(128, return_sequences=True, 
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                  input_shape=(window_size, 1)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second bidirectional LSTM block
    Bidirectional(LSTM(64, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
    BatchNormalization(),
    Dropout(0.25),
    
    # Third LSTM block
    LSTM(32, return_sequences=False,
         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Dense layers
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

lstm_improved.compile(
    optimizer=Adam(learning_rate=0.0008),
    loss='mse',
    metrics=['mae']
)

checkpoint = ModelCheckpoint(
    'best_lstm_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Train improved model
print("\nTraining Improved LSTM Model...")
history_lstm_improved = lstm_improved.fit(
    X_train_uni_3d, y_train_uni,
    validation_split=0.2,
    epochs=60,  # Nombre exact d'epochs comme dans le notebook
    batch_size=64,
    callbacks=[reduce_lr_improved, checkpoint],  # SANS early_stop_improved
    verbose=1
)

# IMPORTANT: Charger les meilleurs poids sauvegardés par ModelCheckpoint
try:
    if os.path.exists('best_lstm_improved.h5'):
        print("\n📥 Chargement des meilleurs poids LSTM depuis le checkpoint...")
        lstm_improved = load_model('best_lstm_improved.h5', compile=False)
        lstm_improved.compile(optimizer=Adam(learning_rate=0.0008), loss='mse', metrics=['mae'])
        print("✓ Meilleurs poids LSTM chargés")
    else:
        print("⚠️ Fichier best_lstm_improved.h5 non trouvé, utilisation du modèle actuel")
except Exception as e:
    print(f"⚠️ Impossible de charger le checkpoint LSTM: {e}")
    print("   Utilisation du modèle actuel")

# Make predictions
y_pred_lstm_improved = lstm_improved.predict(X_test_uni_3d, verbose=0).flatten()

# Evaluate (les métriques seront calculées automatiquement)
results_lstm_improved = evaluate_model(y_test_uni, y_pred_lstm_improved, "LSTM (Improved)")
print(f"✓ LSTM (Improved) - RMSE: {results_lstm_improved['RMSE']:.3f}, MAE: {results_lstm_improved['MAE']:.3f}, R²: {results_lstm_improved['R2']:.4f}")

# ============================================================================
# 8. LSTM IMPROVED MULTIVARIATE (identique au notebook Cell 59)
# ============================================================================
print("\n7. Entraînement LSTM (Improved Multivariate)...")

lstm_multi_improved = Sequential([
    # First bidirectional LSTM block
    Bidirectional(LSTM(128, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                  input_shape=(window_size, n_features)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second bidirectional LSTM block
    Bidirectional(LSTM(64, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
    BatchNormalization(),
    Dropout(0.25),
    
    # Third LSTM block
    LSTM(32, return_sequences=False,
         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Dense layers with feature integration
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
    optimizer=Adam(learning_rate=0.0008),
    loss='mse',
    metrics=['mae']
)

checkpoint = ModelCheckpoint(
    'best_lstm_multi_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Train improved model
print("\nTraining Improved Multivariate LSTM Model...")
history_lstm_multi_improved = lstm_multi_improved.fit(
    X_train_multi_3d, y_train_multi,
    validation_split=0.2,
    epochs=80,  # Nombre exact d'epochs comme dans le notebook
    batch_size=64,
    callbacks=[reduce_lr_improved, checkpoint],  # SANS early_stop_improved
    verbose=1
)

# IMPORTANT: Charger les meilleurs poids sauvegardés par ModelCheckpoint
try:
    if os.path.exists('best_lstm_multi_improved.h5'):
        print("\n📥 Chargement des meilleurs poids LSTM Multivariate depuis le checkpoint...")
        lstm_multi_improved = load_model('best_lstm_multi_improved.h5', compile=False)
        lstm_multi_improved.compile(optimizer=Adam(learning_rate=0.0008), loss='mse', metrics=['mae'])
        print("✓ Meilleurs poids LSTM Multivariate chargés")
    else:
        print("⚠️ Fichier best_lstm_multi_improved.h5 non trouvé, utilisation du modèle actuel")
except Exception as e:
    print(f"⚠️ Impossible de charger le checkpoint LSTM Multivariate: {e}")
    print("   Utilisation du modèle actuel")

# Make predictions
y_pred_lstm_multi_improved = lstm_multi_improved.predict(X_test_multi_3d, verbose=0).flatten()

# Evaluate (les métriques seront calculées automatiquement)
results_lstm_multi_improved = evaluate_model(y_test_multi, y_pred_lstm_multi_improved, "Multivariate LSTM (Improved)")
print(f"✓ LSTM (Multivariate Improved) - RMSE: {results_lstm_multi_improved['RMSE']:.3f}, MAE: {results_lstm_multi_improved['MAE']:.3f}, R²: {results_lstm_multi_improved['R2']:.4f}")

# ============================================================================
# 9. SAUVEGARDE DES MODÈLES ET MÉTRIQUES
# ============================================================================
print("\n8. Sauvegarde des modèles...")

# Créer le dossier models
os.makedirs('models', exist_ok=True)

# Sauvegarder le scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_all, f)

# Sauvegarder les paramètres
params = {
    'window_size': window_size,
    'n_features': n_features,
    'feature_cols': feature_cols,
    'target_col': target_col
}
with open('models/params.pkl', 'wb') as f:
    pickle.dump(params, f)

# Sauvegarder Decision Tree
with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(tree, f)

# Sauvegarder les modèles Keras
mlp_improved.save('models/mlp_model.h5')
cnn_improved.save('models/cnn_model.h5')
lstm_improved.save('models/lstm_uni_model.h5')
lstm_multi_improved.save('models/lstm_multi_model.h5')

# Sauvegarder les métriques
model_metrics = {
    'Decision Tree': {
        'RMSE': results_tree['RMSE'],
        'MAE': results_tree['MAE'],
        'R2': results_tree['R2']
    },
    'MLP': {
        'RMSE': results_mlp_improved['RMSE'],
        'MAE': results_mlp_improved['MAE'],
        'R2': results_mlp_improved['R2']
    },
    'CNN': {
        'RMSE': results_cnn_improved['RMSE'],
        'MAE': results_cnn_improved['MAE'],
        'R2': results_cnn_improved['R2']
    },
    'LSTM (Univariate)': {
        'RMSE': results_lstm_improved['RMSE'],
        'MAE': results_lstm_improved['MAE'],
        'R2': results_lstm_improved['R2']
    },
    'LSTM (Multivariate)': {
        'RMSE': results_lstm_multi_improved['RMSE'],
        'MAE': results_lstm_multi_improved['MAE'],
        'R2': results_lstm_multi_improved['R2']
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

