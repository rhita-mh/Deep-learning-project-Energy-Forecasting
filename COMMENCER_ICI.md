# 🚀 COMMENCEZ ICI - Guide Simple

## ⚡ Ce que vous devez faire MAINTENANT (dans l'ordre)

---

### 1️⃣ Ouvrez Anaconda Prompt
   - Appuyez sur **Windows**
   - Tapez **"Anaconda Prompt"**
   - Cliquez dessus

---

### 2️⃣ Activez l'environnement
```bash
conda activate tf_clean
```
✅ Vous devriez voir `(tf_clean)` au début de la ligne

---

### 3️⃣ Allez dans le dossier
```bash
cd C:\Users\asus\Desktop\cur
```

---

### 4️⃣ Vérifiez TensorFlow
```bash
python -c "import tensorflow as tf; print('OK:', tf.__version__)"
```
✅ Si ça marche → passez à l'étape 5
❌ Si erreur → tapez : `pip install tensorflow`

---

### 5️⃣ Installez les dépendances
```bash
pip install -r requirements.txt
```
⏱️ Attendez 2-5 minutes

---

### 6️⃣ Entraînez les modèles (UNE SEULE FOIS)
```bash
python train_models.py
```
⏱️ **ATTENDEZ 15-30 MINUTES** - Ne fermez pas la fenêtre !

✅ À la fin, vous verrez : `✓ TOUS LES MODÈLES ONT ÉTÉ ENTRÂINÉS`

---

### 7️⃣ Vérifiez les modèles
```bash
dir models
```
✅ Vous devez voir 7 fichiers (.pkl et .h5)

---

### 8️⃣ Lancez l'application
```bash
streamlit run app.py
```
✅ Votre navigateur s'ouvrira automatiquement !

---

## 🎯 Les prochaines fois

Une fois que tout est fait, pour lancer l'app :

```bash
conda activate tf_clean
cd C:\Users\asus\Desktop\cur
streamlit run app.py
```

C'est tout ! 🎉

---

## 📋 Checklist

- [ ] Anaconda Prompt ouvert
- [ ] `conda activate tf_clean` ✅
- [ ] `cd C:\Users\asus\Desktop\cur` ✅
- [ ] TensorFlow fonctionne ✅
- [ ] `pip install -r requirements.txt` ✅
- [ ] `python train_models.py` ✅ (15-30 min)
- [ ] 7 fichiers dans `models/` ✅
- [ ] `streamlit run app.py` ✅
- [ ] Application ouverte dans le navigateur ✅

---

## ❓ Besoin d'aide ?

- Guide détaillé : `GUIDE_COMPLET.md`
- Problèmes : voir la section "Problèmes courants" dans `GUIDE_COMPLET.md`

---

**Commencez par l'étape 1 ! 👆**

