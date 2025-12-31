"""
Test simple pour vérifier l'inversion de la normalisation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("TEST D'INVERSION DE NORMALISATION")
print("=" * 60)

# Charger les données
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)
df = df.drop_duplicates()

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

def invert_scale_target(scaled_target: np.ndarray) -> np.ndarray:
    """Invert MinMax scaling for the target column (assumed at index 0)."""
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    inv = scaler_all.inverse_transform(dummy)
    return inv[:, 0]

# Test 1: Vérifier que l'inversion fonctionne correctement
print("\n1. Test d'inversion:")
print(f"   Nombre de features: {n_features}")

# Prendre quelques valeurs de test
test_consumption_original = test_values[:10, 0]  # 10 premières valeurs de Consumption
test_consumption_scaled = test_scaled[:10, 0]

print(f"\n   Valeurs originales (10 premières):")
print(f"   {test_consumption_original}")

print(f"\n   Valeurs normalisées (10 premières):")
print(f"   {test_consumption_scaled}")

# Inverser
test_consumption_inverted = invert_scale_target(test_scaled[:10, 0])

print(f"\n   Valeurs inversées (10 premières):")
print(f"   {test_consumption_inverted}")

# Vérifier si elles correspondent
diff = np.abs(test_consumption_original - test_consumption_inverted)
print(f"\n   Différences (devrait être ~0):")
print(f"   {diff}")
print(f"   Max différence: {diff.max():.10f}")

if diff.max() < 1e-6:
    print(f"   ✅ L'inversion fonctionne correctement!")
else:
    print(f"   ❌ PROBLÈME: L'inversion ne fonctionne pas correctement!")

# Test 2: Vérifier avec des valeurs dans [0, 1]
print("\n2. Test avec valeurs dans [0, 1]:")
test_scaled_values = np.array([0.0, 0.5, 1.0])
test_inverted = invert_scale_target(test_scaled_values)

print(f"   Valeurs normalisées: {test_scaled_values}")
print(f"   Valeurs inversées: {test_inverted}")

# Vérifier les limites
min_consumption = df['Consumption'].min()
max_consumption = df['Consumption'].max()

print(f"\n   Min Consumption dans les données: {min_consumption:.2f}")
print(f"   Max Consumption dans les données: {max_consumption:.2f}")

if abs(test_inverted[0] - min_consumption) < 1:
    print(f"   ✅ Min correspond (0.0 -> {test_inverted[0]:.2f} ≈ {min_consumption:.2f})")
else:
    print(f"   ❌ Min ne correspond pas (0.0 -> {test_inverted[0]:.2f} ≠ {min_consumption:.2f})")

if abs(test_inverted[2] - max_consumption) < 1:
    print(f"   ✅ Max correspond (1.0 -> {test_inverted[2]:.2f} ≈ {max_consumption:.2f})")
else:
    print(f"   ❌ Max ne correspond pas (1.0 -> {test_inverted[2]:.2f} ≠ {max_consumption:.2f})")

print("\n" + "=" * 60)
print("TEST TERMINÉ")
print("=" * 60)

