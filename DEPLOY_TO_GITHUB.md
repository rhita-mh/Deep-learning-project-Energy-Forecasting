# 🚀 Guide de Déploiement sur GitHub

## 📋 Prérequis

1. ✅ Compte GitHub créé
2. ✅ Git installé (vérifié: `git --version`)
3. ✅ Projet prêt (modèles entraînés)

## 🔧 Étapes de Déploiement

### 1️⃣ Initialiser Git (si pas déjà fait)

```bash
git init
```

### 2️⃣ Vérifier les fichiers à ajouter

```bash
git status
```

### 3️⃣ Ajouter tous les fichiers (sauf ceux dans .gitignore)

```bash
git add .
```

**Note:** Les fichiers suivants seront **exclus automatiquement** (dans .gitignore):
- `models/*.h5` et `models/*.pkl` (trop volumineux)
- `__pycache__/`
- `*.log`
- Fichiers temporaires

### 4️⃣ Créer le premier commit

```bash
git commit -m "Initial commit: AI Energy Forecast System"
```

### 5️⃣ Créer un dépôt sur GitHub

1. Allez sur [GitHub.com](https://github.com)
2. Cliquez sur **"+"** en haut à droite → **"New repository"**
3. Nommez le dépôt (ex: `ai-energy-forecast`)
4. **Ne cochez PAS** "Initialize with README" (on a déjà un README)
5. Cliquez sur **"Create repository"**

### 6️⃣ Lier le dépôt local à GitHub

```bash
git remote add origin https://github.com/VOTRE_USERNAME/ai-energy-forecast.git
```

**Remplacez `VOTRE_USERNAME` par votre nom d'utilisateur GitHub**

### 7️⃣ Pousser le code sur GitHub

```bash
git branch -M main
git push -u origin main
```

**Note:** Si GitHub vous demande vos identifiants:
- Utilisez un **Personal Access Token** (pas votre mot de passe)
- Créez-en un: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

## 📝 Fichiers Inclus dans le Dépôt

✅ **Inclus:**
- `app.py` - Application Streamlit principale
- `train_models.py` - Script d'entraînement
- `requirements.txt` - Dépendances
- `README.md` - Documentation
- Tous les fichiers `.md` de documentation
- Scripts Python (diagnostic, test, etc.)
- Fichiers batch/shell pour Windows/Linux

❌ **Exclus (trop volumineux):**
- `models/*.h5` - Modèles entraînés
- `models/*.pkl` - Scaler et paramètres
- `best_*.h5` - Fichiers temporaires d'entraînement

## 🔄 Mise à Jour du Dépôt

Après avoir modifié des fichiers:

```bash
git add .
git commit -m "Description des modifications"
git push
```

## 📦 Ajouter les Modèles (Optionnel)

Si vous voulez inclure les modèles entraînés (attention: fichiers volumineux):

1. Utilisez **Git LFS** (Large File Storage):
```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.pkl"
git add .gitattributes
git add models/
git commit -m "Add trained models with Git LFS"
git push
```

2. Ou utilisez des **releases GitHub** pour les modèles

## 🌐 Déploiement sur Streamlit Cloud (Optionnel)

1. Allez sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez votre compte GitHub
3. Sélectionnez votre dépôt
4. Configurez:
   - **Main file:** `app.py`
   - **Python version:** 3.8+
5. Ajoutez les secrets si nécessaire
6. Déployez!

**Note:** Pour Streamlit Cloud, vous devrez:
- Ajouter les modèles au dépôt (via Git LFS ou releases)
- Ou entraîner les modèles lors du déploiement

## ✅ Checklist de Déploiement

- [ ] Git initialisé
- [ ] `.gitignore` vérifié
- [ ] Fichiers ajoutés (`git add .`)
- [ ] Premier commit créé
- [ ] Dépôt GitHub créé
- [ ] Remote ajouté
- [ ] Code poussé sur GitHub
- [ ] README.md à jour
- [ ] Requirements.txt complet

## 🆘 Problèmes Courants

**Erreur: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/VOTRE_USERNAME/ai-energy-forecast.git
```

**Erreur: "Authentication failed"**
- Utilisez un Personal Access Token au lieu du mot de passe
- Créez-en un: GitHub → Settings → Developer settings → Personal access tokens

**Erreur: "Large files"**
- Les modèles sont trop volumineux
- Utilisez Git LFS ou excluez-les du dépôt

## 📚 Ressources

- [GitHub Docs](https://docs.github.com)
- [Git LFS](https://git-lfs.github.com)
- [Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)

