# 📋 Guide Étape par Étape - AI Energy Forecast

## ✅ ÉTAPE 1 : Vérifier Python

Ouvrez PowerShell ou CMD et vérifiez que Python est installé :

```bash
python --version
```

**Résultat attendu** : Python 3.8 ou supérieur

Si Python n'est pas installé, téléchargez-le depuis [python.org](https://www.python.org/downloads/)

---

## ✅ ÉTAPE 2 : Activer l'environnement tf_clean

**IMPORTANT** : Vous devez utiliser l'environnement virtuel `tf_clean` que vous avez créé !

### Option A : Avec Anaconda Prompt (RECOMMANDÉ)
1. Ouvrez **Anaconda Prompt** depuis le menu Démarrer
2. Activez l'environnement :
```bash
conda activate tf_clean
```
3. Naviguez vers le dossier :
```bash
cd C:\Users\asus\Desktop\cur
```

### Option B : Depuis PowerShell
```bash
conda activate tf_clean
cd C:\Users\asus\Desktop\cur
```

## ✅ ÉTAPE 2b : Installer les dépendances (si nécessaire)

Vérifiez d'abord si TensorFlow est installé :
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

Si TensorFlow n'est pas installé, installez les dépendances :
```bash
pip install -r requirements.txt
```

⏱️ **Temps estimé** : 5-10 minutes

**Ce qui sera installé** :
- streamlit (interface web)
- pandas, numpy (traitement de données)
- scikit-learn (machine learning)
- tensorflow (deep learning)
- plotly (graphiques interactifs)
- et autres...

**✅ Vérification** : Si tout s'est bien passé, vous verrez "Successfully installed..."

---

## ✅ ÉTAPE 3 : Entraîner les modèles (IMPORTANT - Première fois uniquement)

Cette étape va créer les modèles pré-entraînés. C'est la partie la plus longue.

```bash
python train_models.py
```

⏱️ **Temps estimé** : 15-30 minutes (selon votre machine)

**Ce qui va se passer** :
1. ✅ Chargement des données CSV
2. ✅ Préparation et normalisation des données
3. ✅ Entraînement de 5 modèles différents :
   - Decision Tree
   - MLP (réseau de neurones)
   - CNN (réseau convolutif)
   - LSTM Univariate
   - LSTM Multivariate
4. ✅ Sauvegarde des modèles dans le dossier `models/`

**✅ Vérification** : À la fin, vous devriez voir :
```
✓ TOUS LES MODÈLES ONT ÉTÉ ENTRÂINÉS ET SAUVEGARDÉS
```

**⚠️ Note** : Cette étape ne doit être faite qu'une seule fois. Les modèles seront sauvegardés et réutilisables.

---

## ✅ ÉTAPE 4 : Vérifier que les modèles sont créés

Vérifiez que le dossier `models/` contient les fichiers :

```bash
dir models
```

**Fichiers attendus** :
- ✅ `scaler.pkl`
- ✅ `params.pkl`
- ✅ `decision_tree.pkl`
- ✅ `mlp_model.h5`
- ✅ `cnn_model.h5`
- ✅ `lstm_uni_model.h5`
- ✅ `lstm_multi_model.h5`

Si tous ces fichiers sont présents, vous pouvez passer à l'étape suivante !

---

## ✅ ÉTAPE 5 : Lancer l'application web

Il y a **2 façons** de lancer l'application :

### Option A : Avec le script automatique (RECOMMANDÉ)
Double-cliquez simplement sur le fichier **`run_app.bat`**

### Option B : Avec la commande manuelle
```bash
streamlit run app.py
```

**Ce qui va se passer** :
1. L'application va démarrer
2. Votre navigateur s'ouvrira automatiquement
3. L'adresse sera : `http://localhost:8501`

⏱️ **Temps de démarrage** : 10-30 secondes

---

## ✅ ÉTAPE 6 : Utiliser l'interface web

Une fois l'application ouverte dans votre navigateur :

### 1. **Sélectionner un modèle** (dans la sidebar à gauche)
   - Recommandé : **"LSTM (Multivariate)"** pour la meilleure précision
   - Ou **"Decision Tree"** pour des prédictions rapides

### 2. **Choisir le mode de prédiction**
   - **📈 Prédiction en temps réel** : Utilise les dernières 24h de données
   - **📅 Prédiction avec données historiques** : Compare avec les vraies valeurs

### 3. **Générer une prédiction**
   - Cliquez sur le bouton **"🔄 Générer Prédiction"** ou **"🔮 Générer Prédiction"**

### 4. **Visualiser les résultats**
   - Graphiques interactifs
   - Métriques (prédiction, erreur, variation)
   - Comparaisons visuelles

---

## 🎯 Résumé des commandes

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner les modèles (une seule fois)
python train_models.py

# 3. Lancer l'application
streamlit run app.py
```

---

## ⚠️ Problèmes courants et solutions

### ❌ "pip n'est pas reconnu"
**Solution** : Utilisez `python -m pip install -r requirements.txt`

### ❌ "Module not found"
**Solution** : Réinstallez les dépendances : `pip install -r requirements.txt`

### ❌ "Impossible de charger les modèles"
**Solution** : Vous devez d'abord exécuter `python train_models.py`

### ❌ "Port 8501 already in use"
**Solution** : Fermez l'application précédente ou utilisez un autre port :
```bash
streamlit run app.py --server.port 8502
```

### ❌ L'entraînement est très lent
**Solution** : C'est normal ! L'entraînement prend du temps. Laissez-le tourner.

---

## 📞 Besoin d'aide ?

- Consultez `README.md` pour plus de détails
- Consultez `QUICKSTART.md` pour un guide rapide
- Vérifiez que tous les fichiers sont présents dans le dossier

---

## 🎉 C'est tout !

Une fois ces étapes terminées, vous aurez une interface web fonctionnelle pour prédire la consommation électrique en temps réel !

