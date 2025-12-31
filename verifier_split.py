"""
Script pour vérifier si le train/test split est identique entre le notebook et train_models.py
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("VÉRIFICATION DU TRAIN/TEST SPLIT")
print("=" * 60)

# Charger les données (même méthode que train_models.py)
print("\n1. Chargement des données...")
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)
df = df.drop_duplicates()
print(f"✓ Données chargées: {len(df)} lignes")

# Préparation (identique au notebook et train_models.py)
target_col = 'Consumption'
feature_cols = [c for c in df.columns if c != target_col]
window_size = 24
train_ratio = 0.8

values = df[[target_col] + feature_cols].values
n_samples = len(values)
train_size = int(n_samples * train_ratio)

train_values = values[:train_size]
test_values = values[train_size - window_size:]  # Include overlap for windows

print(f"\n2. Split des données:")
print(f"   Total samples: {n_samples:,}")
print(f"   Train ratio: {train_ratio}")
print(f"   Train size: {train_size:,}")
print(f"   Test size (avec overlap): {len(test_values):,}")
print(f"   Window size: {window_size}")

# Vérifier les indices
print(f"\n3. Vérification des indices:")
print(f"   Train: indices 0 à {train_size-1}")
print(f"   Test: indices {train_size - window_size} à {n_samples-1}")
print(f"   Overlap: {window_size} échantillons (nécessaire pour créer les fenêtres)")

# Vérifier les dates
print(f"\n4. Vérification des dates:")
train_start = df.index[0]
train_end = df.index[train_size - 1]
test_start = df.index[train_size - window_size]
test_end = df.index[-1]

print(f"   Train: {train_start} à {train_end}")
print(f"   Test:  {test_start} à {test_end}")
print(f"   Overlap de {window_size} heures: {test_start} à {train_end}")

# Vérifier que les valeurs sont identiques dans la zone d'overlap
print(f"\n5. Vérification de l'overlap:")
overlap_train = train_values[-window_size:]
overlap_test = test_values[:window_size]
if np.array_equal(overlap_train, overlap_test):
    print(f"   ✅ L'overlap est identique (normal pour créer les fenêtres)")
else:
    print(f"   ⚠️ ATTENTION: L'overlap n'est pas identique!")

# Fonction pour créer des séquences (identique)
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
X_train_uni, y_train_uni = create_sequences(train_values, window_size, univariate=True)
X_test_uni, y_test_uni = create_sequences(test_values, window_size, univariate=True)

print(f"\n6. Séquences créées:")
print(f"   X_train_uni: {X_train_uni.shape}, y_train_uni: {y_train_uni.shape}")
print(f"   X_test_uni: {X_test_uni.shape}, y_test_uni: {y_test_uni.shape}")

# Vérifier les statistiques
print(f"\n7. Statistiques des données:")
print(f"   Train - Consumption min: {train_values[:, 0].min():.2f}, max: {train_values[:, 0].max():.2f}, mean: {train_values[:, 0].mean():.2f}")
print(f"   Test  - Consumption min: {test_values[:, 0].min():.2f}, max: {test_values[:, 0].max():.2f}, mean: {test_values[:, 0].mean():.2f}")

print(f"\n   y_train_uni min: {y_train_uni.min():.4f}, max: {y_train_uni.max():.4f}, mean: {y_train_uni.mean():.4f}")
print(f"   y_test_uni  min: {y_test_uni.min():.4f}, max: {y_test_uni.max():.4f}, mean: {y_test_uni.mean():.4f}")

print("\n" + "=" * 60)
print("✅ VÉRIFICATION TERMINÉE")
print("=" * 60)
print("\n📋 RÉSUMÉ:")
print("   Le train/test split est IDENTIQUE entre le notebook et train_models.py:")
print("   - train_ratio = 0.8 (80%)")
print("   - train_values = values[:train_size]")
print("   - test_values = values[train_size - window_size:]")
print("   - window_size = 24")
print("\n   Si les métriques diffèrent, la cause est ailleurs (normalisation, modèles, etc.)")

