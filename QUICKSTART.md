# 🚀 Guide de Démarrage Rapide

## Installation en 3 étapes

### 1️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2️⃣ Entraîner les modèles (première fois uniquement)
```bash
python train_models.py
```
⏱️ **Temps estimé**: 10-30 minutes selon votre machine

### 3️⃣ Lancer l'application
```bash
streamlit run app.py
```

Ou utilisez les scripts de démarrage:
- **Windows**: Double-cliquez sur `run_app.bat`
- **Linux/Mac**: `bash run_app.sh`

## 🎯 Utilisation Rapide

1. **Ouvrez l'application** dans votre navigateur (généralement `http://localhost:8501`)

2. **Sélectionnez un modèle** dans la sidebar (recommandé: LSTM Multivariate)

3. **Choisissez le mode**:
   - **Prédiction en temps réel**: Utilise les dernières 24h
   - **Prédiction historique**: Compare avec les vraies valeurs

4. **Cliquez sur "Générer Prédiction"** et visualisez les résultats!

## 📊 Modèles Disponibles

| Modèle | Rapidité | Précision | Recommandé pour |
|--------|----------|-----------|----------------|
| Decision Tree | ⚡⚡⚡ | ⭐⭐⭐ | Démonstration rapide |
| MLP | ⚡⚡ | ⭐⭐⭐⭐ | Équilibre vitesse/précision |
| CNN | ⚡⚡ | ⭐⭐⭐⭐ | Patterns complexes |
| LSTM (Uni) | ⚡ | ⭐⭐⭐⭐⭐ | Meilleure précision |
| LSTM (Multi) | ⚡ | ⭐⭐⭐⭐⭐ | **Meilleur choix** |

## ⚠️ Dépannage Rapide

**"Impossible de charger les modèles"**
→ Exécutez `python train_models.py` d'abord

**"Module not found"**
→ `pip install -r requirements.txt`

**L'application ne démarre pas**
→ Vérifiez que le port 8501 est libre

## 💡 Astuces

- Les prédictions sont plus précises avec des données récentes
- Le modèle LSTM Multivariate utilise toutes les sources d'énergie
- Utilisez le mode historique pour évaluer la précision

## 📞 Besoin d'aide?

Consultez le fichier `README.md` pour plus de détails.

