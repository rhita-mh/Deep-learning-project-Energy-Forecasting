# 📋 Guide Étape par Étape - Calculer les Métriques Réelles

## 🎯 Objectif
Calculer les vraies métriques (RMSE, MAE, R²) de tous les modèles sur le test set pour remplacer les estimations.

---

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

## ✅ ÉTAPE 4 : Vérifier que les modèles existent

Tapez :

```bash
dir models
```

**Fichiers attendus** (7 fichiers) :
- ✅ `scaler.pkl`
- ✅ `params.pkl`
- ✅ `decision_tree.pkl`
- ✅ `mlp_model.h5`
- ✅ `cnn_model.h5`
- ✅ `lstm_uni_model.h5`
- ✅ `lstm_multi_model.h5`

**Si tous ces fichiers sont présents** → Passez à l'étape 5
**Si des fichiers manquent** → Vous devrez ré-entraîner (étape 5 prendra plus de temps)

---

## ✅ ÉTAPE 5 : Exécuter le script d'entraînement

Tapez :

```bash
python train_models.py
```

**Ce qui va se passer** :
1. ✅ Chargement des données
2. ✅ Préparation des données
3. ✅ Entraînement des modèles (si nécessaire)
4. ✅ **NOUVEAU** : Calcul des métriques réelles sur le test set
5. ✅ Sauvegarde des métriques dans `models/model_metrics.pkl`

⏱️ **Temps estimé** :
- Si les modèles existent déjà : **5-10 minutes** (juste le calcul des métriques)
- Si les modèles n'existent pas : **15-30 minutes** (entraînement complet)

**✅ Vérification** : À la fin, vous devriez voir :
```
Métriques calculées sur le test set:
============================================================
Decision Tree:
  RMSE: XXX.XX
  MAE:  XXX.XX
  R²:   X.XXXX
...
============================================================
✓ TOUS LES MODÈLES ONT ÉTÉ ENTRÂINÉS ET SAUVEGARDÉS
✓ MÉTRIQUES CALCULÉES ET SAUVEGARDÉES
```

---

## ✅ ÉTAPE 6 : Vérifier que les métriques sont sauvegardées

Tapez :

```bash
dir models
```

**Nouveau fichier attendu** :
- ✅ `model_metrics.pkl` (nouveau fichier)

Si ce fichier est présent, les métriques réelles ont été calculées !

---

## ✅ ÉTAPE 7 : Redémarrer l'application Streamlit

**Option A : Si l'application est déjà ouverte**
1. Dans la fenêtre où Streamlit tourne, appuyez sur **Ctrl+C** pour l'arrêter
2. Puis tapez : `streamlit run app.py`

**Option B : Si l'application n'est pas ouverte**
Tapez simplement :

```bash
streamlit run app.py
```

---

## ✅ ÉTAPE 8 : Vérifier dans l'application

1. **Ouvrez la page "🎯 Performances des Modèles"**
2. **Sélectionnez "LSTM (Multivariate)"** dans Deep Learning Models
3. **Vérifiez les métriques** : Elles devraient maintenant être les vraies valeurs calculées

4. **Allez dans "⚖️ Comparaison des Modèles"**
5. **Lisez la note explicative** qui explique pourquoi une prédiction unique peut différer du RMSE global

---

## 🎯 Résultat Attendu

### Avant (Estimations) :
- LSTM (Multivariate): RMSE: 155.0 (estimation)

### Après (Métriques Réelles) :
- LSTM (Multivariate): RMSE: [valeur réelle calculée sur le test set]

Les métriques seront maintenant **cohérentes** et refléteront la vraie performance de chaque modèle.

---

## ⚠️ Notes Importantes

1. **Une seule prédiction ≠ Performance globale**
   - Le RMSE est calculé sur 10,000+ prédictions
   - Une date spécifique peut avoir une erreur différente
   - C'est normal qu'un modèle excellent en moyenne ait parfois de mauvaises prédictions

2. **Les modèles LSTM sont généralement meilleurs**
   - Ils ont le meilleur RMSE global
   - Mais peuvent avoir des erreurs sur certaines dates spécifiques
   - C'est pourquoi vous voyez parfois une mauvaise prédiction sur une date

3. **Pour voir la vraie performance** :
   - Consultez la page "Performances des Modèles" (métriques globales)
   - La page "Comparaison avec Valeurs Réelles" montre seulement une prédiction unique

---

## 📝 Commandes Résumées

```bash
# 1. Activer l'environnement
conda activate tf_clean

# 2. Aller dans le dossier
cd C:\Users\asus\Desktop\cur

# 3. Calculer les métriques (et entraîner si nécessaire)
python train_models.py

# 4. Lancer l'application
streamlit run app.py
```

---

## ❓ Problèmes Possibles

### ❌ "FileNotFoundError: models/scaler.pkl"
**Solution** : Les modèles n'existent pas. L'étape 5 va les créer (cela prendra 15-30 minutes).

### ❌ "Module not found"
**Solution** : Vérifiez que vous êtes dans l'environnement tf_clean :
```bash
conda activate tf_clean
pip install -r requirements.txt
```

### ❌ L'application ne charge pas les nouvelles métriques
**Solution** : 
1. Arrêtez l'application (Ctrl+C)
2. Supprimez le cache : `rm -rf .streamlit/cache` (ou supprimez le dossier manuellement)
3. Relancez : `streamlit run app.py`

---

**C'est tout ! Suivez ces étapes dans l'ordre et vous aurez les métriques réelles ! 🚀**

