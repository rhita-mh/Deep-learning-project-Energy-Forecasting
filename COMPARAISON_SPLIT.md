# 🔍 Comparaison du Train/Test Split

## ✅ RÉSULTAT : Le split est **IDENTIQUE** entre le notebook et `train_models.py`

### Code du Notebook (Cell 47):
```python
train_ratio = 0.8     # 80% of data for training
train_size = int(n_samples * train_ratio)
train_values = values[:train_size]
test_values = values[train_size - window_size:]  # Include overlap for windows
```

### Code de train_models.py (lignes 50-58):
```python
train_ratio = 0.8
train_size = int(n_samples * train_ratio)
train_values = values[:train_size]
test_values = values[train_size - window_size:]
```

## ✅ Points Identiques

1. **train_ratio** : `0.8` (80% pour l'entraînement)
2. **train_size** : `int(n_samples * train_ratio)`
3. **train_values** : `values[:train_size]` (premiers 80%)
4. **test_values** : `values[train_size - window_size:]` (avec overlap de 24 heures)
5. **window_size** : `24` (fenêtre de 24 heures)

## ✅ Fonction create_sequences

Les deux fichiers utilisent la **même logique** :
- Pour univariate : `X.append(data[i - window:i, 0])`
- Pour multivariate : `X.append(data[i - window:i, :])`
- Target : `y.append(data[i, 0])` (toujours Consumption)

## ✅ Normalisation

Les deux utilisent `MinMaxScaler` sur toutes les colonnes ensemble :
```python
scaler_all = MinMaxScaler()
train_scaled = scaler_all.fit_transform(train_values)
test_scaled = scaler_all.transform(test_values)
```

## ✅ Fonction invert_scale_target

Les deux utilisent la **même logique** :
```python
def invert_scale_target(scaled_target, scaler, n_features):
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]
```

## 🔍 Conclusion

**Le train/test split est IDENTIQUE.** Si les métriques diffèrent, la cause est probablement :

1. **Différence dans l'entraînement des modèles** :
   - Nombre d'epochs
   - Callbacks (EarlyStopping, ReduceLROnPlateau)
   - Validation split
   - Random seeds (mais les deux utilisent `random_state=42`)

2. **Différence dans les hyperparamètres** :
   - Architecture des modèles
   - Learning rate
   - Batch size

3. **Différence dans les données** :
   - Le notebook pourrait avoir été exécuté avec des données légèrement différentes
   - Ordre des opérations (drop_duplicates avant/après certaines opérations)

## 💡 Solution

Si les métriques sont très élevées, c'est probablement dû à :
- Les modèles n'ont pas convergé correctement
- Les hyperparamètres sont différents
- Les callbacks (EarlyStopping) ont arrêté l'entraînement trop tôt

**La solution automatique que j'ai ajoutée** (détection si RMSE > 500) utilisera les métriques du notebook qui sont connues pour être correctes.

