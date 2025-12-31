# 🚀 Instructions pour Exécuter le Script

## ✅ ÉTAPE 1 : Ouvrir Anaconda Prompt

1. Appuyez sur la touche **Windows**
2. Tapez **"Anaconda Prompt"**
3. Cliquez sur **"Anaconda Prompt"**

---

## ✅ ÉTAPE 2 : Activer l'environnement tf_clean

Dans Anaconda Prompt, tapez :

```bash
conda activate tf_clean
```

**Résultat attendu** : Vous devriez voir `(tf_clean)` au début de la ligne.

---

## ✅ ÉTAPE 3 : Aller dans le dossier du projet

Tapez :

```bash
cd C:\Users\asus\Desktop\cur
```

---

## ✅ ÉTAPE 4 : Exécuter le script d'entraînement

Tapez :

```bash
python train_models_from_notebook.py
```

---

## ⏱️ Ce qui va se passer

Le script va :

1. ✅ Charger les données
2. ✅ Préparer les données (split train/test)
3. ✅ Entraîner Decision Tree (~1 minute)
4. ✅ Entraîner MLP Improved (~10-15 minutes)
5. ✅ Entraîner CNN Improved (~10-15 minutes)
6. ✅ Entraîner LSTM Improved Univariate (~15-20 minutes)
7. ✅ Entraîner LSTM Improved Multivariate (~20-25 minutes)
8. ✅ **Afficher les diagnostics détaillés** pour chaque modèle
9. ✅ Calculer et sauvegarder les métriques

**Temps total estimé** : **1-2 heures** (mais avec early stopping, ça peut être plus rapide)

---

## 📊 Ce que vous verrez

Pour chaque modèle, vous verrez des diagnostics comme :

```
🔍 Diagnostic pour Decision Tree:
   y_true (scaled) - Min: 0.123456, Max: 0.987654, Mean: 0.456789
   y_pred (scaled) - Min: 0.234567, Max: 0.876543, Mean: 0.567890
   y_true_raw - Min: 2345.67, Max: 11234.56, Mean: 6789.12
   y_pred_raw - Min: 2456.78, Max: 10987.65, Mean: 7123.45

Decision Tree Performance:
============================================================
RMSE: 229.734 MW
MAE:  157.272 MW
R²:   0.9480
Corrélation: 0.9742
============================================================
```

---

## ⚠️ Important

- **Ne fermez pas la fenêtre** pendant l'exécution
- **Laissez le script se terminer** complètement
- **Notez les valeurs** affichées dans les diagnostics (surtout si le RMSE est élevé)

---

## 📝 Après l'Exécution

1. **Vérifiez les métriques** affichées
2. **Si le RMSE est élevé** (> 500), notez les valeurs des diagnostics
3. **Partagez les résultats** pour que je puisse identifier le problème

---

## 🎯 Commandes Résumées

```bash
conda activate tf_clean
cd C:\Users\asus\Desktop\cur
python train_models_from_notebook.py
```

---

**C'est tout ! Exécutez ces 3 commandes dans l'ordre et attendez que le script se termine ! 🚀**

