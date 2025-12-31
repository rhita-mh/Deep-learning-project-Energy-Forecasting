"""
Script de diagnostic pour vérifier pourquoi les métriques sont élevées
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 60)
print("DIAGNOSTIC DES MÉTRIQUES")
print("=" * 60)

# Charger les données
print("\n1. Chargement des données...")
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)
df = df.drop_duplicates()
print(f"✓ Données chargées: {len(df)} lignes")

# Préparation des données (identique à train_models.py)
target_col = 'Consumption'
feature_cols = [c for c in df.columns if c != target_col]
window_size = 24
train_ratio = 0.8

values = df[[target_col] + feature_cols].values
n_samples = len(values)
train_size = int(n_samples * train_ratio)

train_values = values[:train_size]
test_values = values[train_size - window_size:]

# Normalisation
scaler_all = MinMaxScaler()
train_scaled = scaler_all.fit_transform(train_values)
test_scaled = scaler_all.transform(test_values)

n_features = train_scaled.shape[1]
print(f"✓ Nombre de features: {n_features}")

# Fonction pour créer des séquences
def create_sequences(data, window, univariate=False):
    X, y = [], []
    for i in range(window, len(data)):
        if univariate:
            X.append(data[i - window:i, 0])
        else:
            X.append(data[i - window:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Créer les séquences
X_test_uni, y_test_uni = create_sequences(test_scaled, window_size, univariate=True)
X_test_multi, y_test_multi = create_sequences(test_scaled, window_size, univariate=False)

print(f"\n✓ Séquences de test créées:")
print(f"  X_test_uni: {X_test_uni.shape}, y_test_uni: {y_test_uni.shape}")
print(f"  X_test_multi: {X_test_multi.shape}, y_test_multi: {y_test_multi.shape}")

# Fonction pour inverser la normalisation
def invert_scale_target(scaled_target, scaler, n_features):
    """Inverse la normalisation pour la colonne cible"""
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

# Inverser y_test pour vérification
y_test_raw = invert_scale_target(y_test_uni, scaler_all, n_features)

print(f"\n✓ Statistiques de y_test_raw (après inversion):")
print(f"  Min: {y_test_raw.min():.2f} MW")
print(f"  Max: {y_test_raw.max():.2f} MW")
print(f"  Mean: {y_test_raw.mean():.2f} MW")
print(f"  Std: {y_test_raw.std():.2f} MW")
print(f"  Shape: {y_test_raw.shape}")

# Charger les modèles et faire des prédictions
print("\n2. Chargement des modèles et prédictions...")

try:
    # Charger Decision Tree
    with open('models/decision_tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    
    y_pred_tree = tree.predict(X_test_uni)
    y_pred_tree_raw = invert_scale_target(y_pred_tree, scaler_all, n_features)
    
    print(f"\n✓ Decision Tree - Prédictions:")
    print(f"  y_pred_tree (scaled) - Min: {y_pred_tree.min():.4f}, Max: {y_pred_tree.max():.4f}, Mean: {y_pred_tree.mean():.4f}")
    print(f"  y_pred_tree_raw - Min: {y_pred_tree_raw.min():.2f}, Max: {y_pred_tree_raw.max():.2f}, Mean: {y_pred_tree_raw.mean():.2f}")
    
    # Calculer les métriques
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_tree_raw))
    mae = mean_absolute_error(y_test_raw, y_pred_tree_raw)
    r2 = r2_score(y_test_raw, y_pred_tree_raw)
    
    print(f"\n📊 Métriques Decision Tree:")
    print(f"  RMSE: {rmse:.3f} MW")
    print(f"  MAE:  {mae:.3f} MW")
    print(f"  R²:   {r2:.4f}")
    
    # Comparaison avec le notebook
    print(f"\n📋 Comparaison avec le notebook:")
    print(f"  Notebook - RMSE: 229.734, MAE: 157.272, R²: 0.948")
    print(f"  Calculé  - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
    print(f"  Différence RMSE: {abs(rmse - 229.734):.3f} MW")
    
    # Vérifier les erreurs
    errors = y_test_raw - y_pred_tree_raw
    print(f"\n🔍 Analyse des erreurs:")
    print(f"  Erreur moyenne: {errors.mean():.2f} MW")
    print(f"  Erreur std: {errors.std():.2f} MW")
    print(f"  Erreur min: {errors.min():.2f} MW")
    print(f"  Erreur max: {errors.max():.2f} MW")
    
    # Vérifier si les prédictions sont dans une plage raisonnable
    print(f"\n✅ Vérification de cohérence:")
    print(f"  y_test_raw range: [{y_test_raw.min():.0f}, {y_test_raw.max():.0f}] MW")
    print(f"  y_pred_tree_raw range: [{y_pred_tree_raw.min():.0f}, {y_pred_tree_raw.max():.0f}] MW")
    
    if y_pred_tree_raw.min() < 0:
        print(f"  ⚠️ ATTENTION: Prédictions négatives détectées!")
    if y_pred_tree_raw.max() > 100000:
        print(f"  ⚠️ ATTENTION: Prédictions très élevées détectées!")
    
except Exception as e:
    print(f"❌ Erreur: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINÉ")
print("=" * 60)

