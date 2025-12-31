"""
Script de diagnostic pour comprendre pourquoi le RMSE est élevé
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 60)
print("DIAGNOSTIC RMSE ÉLEVÉ")
print("=" * 60)

# Charger les données
print("\n1. Chargement des données...")
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)
df = df.drop_duplicates()
print(f"✓ Données chargées: {len(df)} lignes")

# Préparation des données
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

def create_sequences(data, window, univariate=False):
    X, y = [], []
    for i in range(window, len(data)):
        if univariate:
            X.append(data[i - window:i, 0])
        else:
            X.append(data[i - window:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_test_uni, y_test_uni = create_sequences(test_scaled, window_size, univariate=True)

def invert_scale_target(scaled_target, scaler, n_features):
    """Inverse la normalisation pour la colonne cible"""
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

# Inverser y_test
y_test_raw = invert_scale_target(y_test_uni, scaler_all, n_features)

print(f"\n2. Statistiques de y_test_raw (valeurs réelles):")
print(f"   Min: {y_test_raw.min():.2f} MW")
print(f"   Max: {y_test_raw.max():.2f} MW")
print(f"   Mean: {y_test_raw.mean():.2f} MW")
print(f"   Std: {y_test_raw.std():.2f} MW")
print(f"   Shape: {y_test_raw.shape}")

# Charger un modèle et vérifier
print("\n3. Test avec Decision Tree...")
try:
    with open('models/decision_tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    
    y_pred_tree = tree.predict(X_test_uni)
    print(f"\n   y_pred_tree (scaled) - Min: {y_pred_tree.min():.4f}, Max: {y_pred_tree.max():.4f}, Mean: {y_pred_tree.mean():.4f}")
    
    # Vérifier si les prédictions sont dans [0, 1] (normalisées)
    if y_pred_tree.min() < 0 or y_pred_tree.max() > 1:
        print(f"   ⚠️ ATTENTION: Les prédictions ne sont PAS dans [0, 1]!")
        print(f"      Cela suggère un problème avec le modèle ou les données d'entrée")
    
    y_pred_tree_raw = invert_scale_target(y_pred_tree, scaler_all, n_features)
    print(f"\n   y_pred_tree_raw - Min: {y_pred_tree_raw.min():.2f}, Max: {y_pred_tree_raw.max():.2f}, Mean: {y_pred_tree_raw.mean():.2f}")
    
    # Vérifier si les prédictions sont dans une plage raisonnable
    consumption_min = df['Consumption'].min()
    consumption_max = df['Consumption'].max()
    print(f"\n   Plage réelle de Consumption: [{consumption_min:.2f}, {consumption_max:.2f}] MW")
    
    if y_pred_tree_raw.min() < consumption_min * 0.5 or y_pred_tree_raw.max() > consumption_max * 1.5:
        print(f"   ⚠️ ATTENTION: Les prédictions sont HORS de la plage normale!")
    
    # Calculer les métriques
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_tree_raw))
    mae = mean_absolute_error(y_test_raw, y_pred_tree_raw)
    r2 = r2_score(y_test_raw, y_pred_tree_raw)
    
    print(f"\n   Métriques calculées:")
    print(f"   RMSE: {rmse:.3f} MW")
    print(f"   MAE:  {mae:.3f} MW")
    print(f"   R²:   {r2:.4f}")
    
    # Comparer avec la valeur attendue
    print(f"\n   Comparaison avec notebook:")
    print(f"   Notebook - RMSE: 229.734 MW")
    print(f"   Calculé  - RMSE: {rmse:.3f} MW")
    print(f"   Différence: {abs(rmse - 229.734):.3f} MW")
    
    if rmse > 500:
        print(f"\n   ❌ PROBLÈME: RMSE très élevé ({rmse:.3f} > 500)")
        print(f"      Causes possibles:")
        print(f"      1. Les prédictions sont complètement fausses")
        print(f"      2. Problème avec l'inversion de la normalisation")
        print(f"      3. Les données de test ne correspondent pas")
        print(f"      4. Le modèle n'a pas été entraîné correctement")
    
    # Analyser les erreurs
    errors = y_test_raw - y_pred_tree_raw
    print(f"\n   Analyse des erreurs:")
    print(f"   Erreur moyenne: {errors.mean():.2f} MW")
    print(f"   Erreur std: {errors.std():.2f} MW")
    print(f"   Erreur min: {errors.min():.2f} MW")
    print(f"   Erreur max: {errors.max():.2f} MW")
    print(f"   Erreur absolue moyenne: {np.abs(errors).mean():.2f} MW")
    
    # Vérifier quelques exemples
    print(f"\n   Exemples (premiers 5):")
    for i in range(min(5, len(y_test_raw))):
        print(f"   Réel: {y_test_raw[i]:.2f} MW, Prédit: {y_pred_tree_raw[i]:.2f} MW, Erreur: {errors[i]:.2f} MW")
    
    # Vérifier la corrélation
    correlation = np.corrcoef(y_test_raw, y_pred_tree_raw)[0, 1]
    print(f"\n   Corrélation entre réel et prédit: {correlation:.4f}")
    if correlation < 0.5:
        print(f"   ⚠️ ATTENTION: Corrélation faible - le modèle ne prédit pas bien")
    
except Exception as e:
    print(f"❌ Erreur: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINÉ")
print("=" * 60)

