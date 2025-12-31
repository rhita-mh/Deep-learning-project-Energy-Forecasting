# 🔍 Diagnostic RMSE Élevé

## Problème
Le RMSE calculé est toujours très élevé même après avoir utilisé le code exact du notebook.

## Causes Possibles

### 1. **Modèles mal entraînés**
- Les modèles n'ont pas convergé
- Early stopping a arrêté trop tôt
- Les poids ne sont pas sauvegardés correctement

### 2. **Problème avec l'inversion de normalisation**
- La fonction `invert_scale_target` ne fonctionne pas correctement
- Le scaler n'est pas le bon
- Les dimensions ne correspondent pas

### 3. **Données incorrectes**
- Les données de test ne correspondent pas
- Le split train/test est différent
- Les séquences ne sont pas créées correctement

### 4. **Problème avec les prédictions**
- Les modèles prédisent des valeurs dans la mauvaise plage
- Les prédictions sont toutes identiques (modèle non entraîné)
- Les prédictions sont inversées

## 🔧 Solutions

### Solution 1 : Vérifier les Diagnostics

Le script `train_models_from_notebook.py` a maintenant des diagnostics détaillés. Quand vous l'exécutez, vous verrez :

```
🔍 Diagnostic pour Decision Tree:
   y_true (scaled) - Min: X.XXXXXX, Max: X.XXXXXX, Mean: X.XXXXXX
   y_pred (scaled) - Min: X.XXXXXX, Max: X.XXXXXX, Mean: X.XXXXXX
   y_true_raw - Min: XXX.XX, Max: XXX.XX, Mean: XXX.XX
   y_pred_raw - Min: XXX.XX, Max: XXX.XX, Mean: XXX.XX
```

**Vérifiez :**
- ✅ `y_pred (scaled)` devrait être dans [0, 1]
- ✅ `y_pred_raw` devrait être dans la plage de Consumption (environ [2000, 12000] MW)
- ✅ La corrélation devrait être > 0.5

### Solution 2 : Vérifier que les Modèles sont Bien Entraînés

**Pour Decision Tree :**
- Le RMSE devrait être ~229.734 MW
- Si c'est beaucoup plus élevé, le modèle n'est pas bien entraîné

**Pour les modèles Deep Learning :**
- Vérifiez les logs d'entraînement
- Le `val_loss` devrait diminuer
- Le modèle ne devrait pas s'arrêter trop tôt (patience=15)

### Solution 3 : Vérifier l'Inversion de Normalisation

Exécutez :
```bash
python test_inversion.py
```

Cela devrait montrer que l'inversion fonctionne correctement.

### Solution 4 : Comparer avec le Notebook

1. **Exécutez le notebook** et notez les métriques exactes
2. **Exécutez le script** et comparez
3. Si les métriques diffèrent, vérifiez :
   - Les mêmes données sont utilisées
   - Les mêmes hyperparamètres
   - Les mêmes callbacks

## 📊 Valeurs Attendues

### Decision Tree
- RMSE: ~229.734 MW
- MAE: ~157.272 MW
- R²: ~0.948

### MLP Improved
- RMSE: ~150-200 MW (à vérifier dans le notebook)
- MAE: ~100-150 MW
- R²: > 0.95

### CNN Improved
- RMSE: ~140-190 MW (à vérifier dans le notebook)
- MAE: ~90-140 MW
- R²: > 0.96

### LSTM Improved
- RMSE: ~130-180 MW (à vérifier dans le notebook)
- MAE: ~80-130 MW
- R²: > 0.97

## 🚨 Si le RMSE est > 1000

Cela indique un problème majeur :
1. **Vérifiez les prédictions** : Sont-elles toutes identiques ? Sont-elles dans la bonne plage ?
2. **Vérifiez l'entraînement** : Les modèles ont-ils vraiment été entraînés ?
3. **Vérifiez les données** : Les données de test sont-elles correctes ?

## 💡 Action Immédiate

1. **Exécutez le script avec diagnostics** :
   ```bash
   python train_models_from_notebook.py
   ```

2. **Regardez les diagnostics** pour chaque modèle

3. **Partagez les valeurs** que vous voyez pour :
   - `y_pred (scaled)` - Min, Max, Mean
   - `y_pred_raw` - Min, Max, Mean
   - RMSE, MAE, R²
   - Corrélation

4. **Avec ces informations**, je pourrai identifier le problème exact

## 🔄 Alternative : Utiliser les Métriques du Notebook

Si le problème persiste, on peut temporairement utiliser les métriques exactes du notebook comme valeurs de référence jusqu'à ce que le problème soit résolu.

---

**Exécutez le script et partagez les diagnostics pour que je puisse identifier le problème exact !**

