# 🔍 Vérification des Métriques Élevées

## Problème Identifié

Les métriques calculées sont **très élevées** par rapport au notebook original.

## Causes Possibles

### 1. **Différence dans les données de test**
- Le notebook pourrait utiliser un split différent
- Vérifier que `train_ratio = 0.8` est identique

### 2. **Problème de normalisation**
- L'inversion de la normalisation pourrait être incorrecte
- Vérifier que `invert_scale_target` fonctionne correctement

### 3. **Modèles mal entraînés**
- Les modèles pourraient ne pas avoir convergé
- Vérifier les logs d'entraînement (loss, val_loss)

### 4. **Différence dans les hyperparamètres**
- Les modèles dans `train_models.py` pourraient avoir des hyperparamètres différents du notebook

## Solution Immédiate

**Option 1 : Utiliser les métriques du notebook (temporaire)**

Si les métriques calculées sont vraiment incorrectes, on peut temporairement utiliser les métriques du notebook qui sont connues pour être correctes.

**Option 2 : Vérifier le calcul**

Exécuter le script de diagnostic pour identifier le problème exact.

## Actions à Prendre

1. **Vérifier les valeurs exactes** : Quelles sont les métriques calculées exactement ?
   - Decision Tree: RMSE = ?
   - MLP: RMSE = ?
   - CNN: RMSE = ?
   - LSTM: RMSE = ?

2. **Comparer avec le notebook** :
   - Decision Tree: RMSE = 229.734 (notebook)
   - MLP: RMSE = ? (notebook - à vérifier)
   - CNN: RMSE = ? (notebook - à vérifier)
   - LSTM: RMSE = ? (notebook - à vérifier)

3. **Si les métriques sont 2-3x plus élevées** :
   - Problème probable : normalisation incorrecte
   - Solution : Vérifier `invert_scale_target`

4. **Si les métriques sont 10x+ plus élevées** :
   - Problème probable : modèles mal entraînés ou données incorrectes
   - Solution : Ré-entraîner les modèles

## Correction Appliquée

J'ai corrigé le calcul pour LSTM Multivariate pour utiliser `y_test_multi_raw` au lieu de `y_test_raw`, même si les valeurs devraient être identiques (les deux représentent la consommation).

## Prochaines Étapes

1. Exécuter `python train_models.py` à nouveau pour recalculer avec la correction
2. Comparer les nouvelles métriques avec le notebook
3. Si toujours élevées, utiliser temporairement les métriques du notebook

