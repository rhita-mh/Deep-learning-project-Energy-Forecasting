# 🔧 Activer l'environnement tf_clean

## Méthode 1 : Avec Anaconda Prompt (RECOMMANDÉ)

1. **Ouvrez Anaconda Prompt** (depuis le menu Démarrer)

2. **Naviguez vers le dossier du projet** :
```bash
cd C:\Users\asus\Desktop\cur
```

3. **Activez l'environnement tf_clean** :
```bash
conda activate tf_clean
```

4. **Vérifiez que TensorFlow est installé** :
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

5. **Installez les dépendances manquantes** (si nécessaire) :
```bash
pip install -r requirements.txt
```

6. **Entraînez les modèles** :
```bash
python train_models.py
```

7. **Lancez l'application** :
```bash
streamlit run app.py
```

---

## Méthode 2 : Depuis PowerShell (si conda est dans le PATH)

```powershell
# Activer l'environnement
conda activate tf_clean

# Aller dans le dossier
cd C:\Users\asus\Desktop\cur

# Vérifier TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Installer les dépendances
pip install -r requirements.txt

# Entraîner les modèles
python train_models.py

# Lancer l'application
streamlit run app.py
```

---

## Méthode 3 : Trouver le chemin de l'environnement

Si vous ne savez pas où se trouve l'environnement, cherchez-le :

```powershell
# Chercher l'environnement (généralement dans Anaconda ou Miniconda)
dir C:\Users\asus\anaconda3\envs\tf_clean
# ou
dir C:\Users\asus\miniconda3\envs\tf_clean
```

Puis activez-le directement :
```powershell
C:\Users\asus\anaconda3\envs\tf_clean\python.exe train_models.py
```

---

## ✅ Vérification rapide

Une fois l'environnement activé, vérifiez que tout est prêt :

```bash
python -c "import tensorflow; import streamlit; import pandas; print('✅ Tous les modules sont installés!')"
```

---

## 📝 Note importante

**Toujours activer l'environnement tf_clean avant d'exécuter les scripts !**

