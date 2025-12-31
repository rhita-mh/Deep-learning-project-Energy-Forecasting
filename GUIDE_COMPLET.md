# 📘 GUIDE COMPLET - AI Energy Forecast System

## 🎯 Ce que vous devez faire (en ordre)

---

## ✅ ÉTAPE 1 : Ouvrir Anaconda Prompt

1. Appuyez sur la touche **Windows**
2. Tapez **"Anaconda Prompt"**
3. Cliquez sur **"Anaconda Prompt"** (ou "Anaconda PowerShell Prompt")

---

## ✅ ÉTAPE 2 : Activer l'environnement tf_clean

Dans Anaconda Prompt, tapez :

```bash
conda activate tf_clean
```

**Résultat attendu** : Vous devriez voir `(tf_clean)` au début de la ligne de commande.

---

## ✅ ÉTAPE 3 : Aller dans le dossier du projet

Tapez :

```bash
cd C:\Users\asus\Desktop\cur
```

---

## ✅ ÉTAPE 4 : Vérifier que TensorFlow est installé

Tapez :

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

**Résultat attendu** : Vous devriez voir quelque chose comme `TensorFlow version: 2.x.x`

**Si erreur** : Installez TensorFlow avec :
```bash
pip install tensorflow
```

---

## ✅ ÉTAPE 5 : Installer les autres dépendances

Tapez :

```bash
pip install -r requirements.txt
```

⏱️ **Temps** : 2-5 minutes

**Ce qui sera installé** :
- streamlit (interface web)
- pandas, numpy (données)
- scikit-learn (machine learning)
- plotly (graphiques)
- matplotlib, seaborn (visualisation)
- statsmodels (analyse statistique)

---

## ✅ ÉTAPE 6 : Entraîner les modèles (IMPORTANT - Première fois uniquement)

Tapez :

```bash
python train_models.py
```

⏱️ **Temps** : 15-30 minutes (selon votre ordinateur)

**Ce qui va se passer** :
1. ✅ Chargement des données (54,170 lignes)
2. ✅ Préparation des données
3. ✅ Entraînement de 5 modèles :
   - Decision Tree
   - MLP (réseau de neurones)
   - CNN (réseau convolutif)
   - LSTM Univariate
   - LSTM Multivariate
4. ✅ Sauvegarde dans le dossier `models/`

**⚠️ IMPORTANT** : 
- Ne fermez pas la fenêtre pendant l'entraînement
- Laissez l'ordinateur travailler
- À la fin, vous verrez : `✓ TOUS LES MODÈLES ONT ÉTÉ ENTRÂINÉS ET SAUVEGARDÉS`

---

## ✅ ÉTAPE 7 : Vérifier que les modèles sont créés

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

Si tous ces fichiers sont là, **bravo !** Les modèles sont prêts.

---

## ✅ ÉTAPE 8 : Lancer l'application web

Tapez :

```bash
streamlit run app.py
```

**Ce qui va se passer** :
1. L'application va démarrer
2. Votre navigateur s'ouvrira automatiquement
3. L'adresse sera : `http://localhost:8501`

⏱️ **Temps de démarrage** : 10-30 secondes

---

## ✅ ÉTAPE 9 : Utiliser l'interface web

Une fois l'application ouverte dans votre navigateur :

### 📊 Dans la sidebar (à gauche) :

1. **Sélectionnez un modèle** :
   - **"LSTM (Multivariate)"** ← Meilleure précision (recommandé)
   - "Decision Tree" ← Rapide pour tester
   - "MLP", "CNN", "LSTM (Univariate)" ← Autres options

2. **Choisissez le mode** :
   - **"📈 Prédiction en temps réel"** ← Utilise les dernières 24h
   - **"📅 Prédiction avec données historiques"** ← Compare avec les vraies valeurs

### 🎯 Pour faire une prédiction :

1. Cliquez sur **"🔄 Générer Prédiction"** (mode temps réel)
   OU
2. Sélectionnez une date/heure puis **"🔮 Générer Prédiction"** (mode historique)

### 📈 Résultats :

Vous verrez :
- **Graphiques interactifs** avec Plotly
- **Métriques** : Prédiction, erreur, variation
- **Comparaisons visuelles** entre prédiction et réalité

---

## 🔄 Les prochaines fois

Une fois que les modèles sont entraînés (ÉTAPE 6), vous n'avez plus besoin de les ré-entraîner !

**Pour lancer l'application** :
1. Ouvrez Anaconda Prompt
2. `conda activate tf_clean`
3. `cd C:\Users\asus\Desktop\cur`
4. `streamlit run app.py`

C'est tout ! 🎉

---

## ⚠️ Problèmes courants

### ❌ "conda n'est pas reconnu"
**Solution** : Utilisez **Anaconda Prompt** au lieu de PowerShell normal

### ❌ "Module not found"
**Solution** : Vérifiez que vous êtes dans l'environnement tf_clean :
```bash
conda activate tf_clean
pip install -r requirements.txt
```

### ❌ "Impossible de charger les modèles"
**Solution** : Vous devez d'abord exécuter l'ÉTAPE 6 (entraîner les modèles)

### ❌ "Port 8501 already in use"
**Solution** : Fermez l'application précédente ou utilisez :
```bash
streamlit run app.py --server.port 8502
```

### ❌ L'entraînement est très lent
**Solution** : C'est normal ! Laissez-le tourner. Cela peut prendre 30 minutes.

---

## 📋 Checklist rapide

- [ ] Anaconda Prompt ouvert
- [ ] Environnement tf_clean activé
- [ ] Dans le dossier `C:\Users\asus\Desktop\cur`
- [ ] TensorFlow installé et fonctionnel
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Modèles entraînés (`python train_models.py`)
- [ ] 7 fichiers dans le dossier `models/`
- [ ] Application lancée (`streamlit run app.py`)
- [ ] Interface web ouverte dans le navigateur

---

## 🎯 Résumé en 3 commandes

Une fois que tout est installé et les modèles entraînés :

```bash
conda activate tf_clean
cd C:\Users\asus\Desktop\cur
streamlit run app.py
```

---

## 📞 Besoin d'aide ?

- Consultez `ACTIVER_ENVIRONNEMENT.md` pour plus de détails sur l'environnement
- Consultez `README.md` pour la documentation complète
- Consultez `QUICKSTART.md` pour un guide rapide

---

**Bonne chance ! 🚀**

