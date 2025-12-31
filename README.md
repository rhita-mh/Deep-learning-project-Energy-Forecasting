# ⚡ AI Energy Forecast System

Interface web moderne avec thème technologique pour la prédiction de consommation électrique en temps réel.

## 🚀 Fonctionnalités

- **5 Modèles de Deep Learning pré-entraînés**:
  - Decision Tree
  - MLP (Multi-Layer Perceptron)
  - CNN (Convolutional Neural Network)
  - LSTM Univariate
  - LSTM Multivariate

- **Prédiction en temps réel**: Utilise les dernières 24 heures pour prédire la consommation future
- **Prédiction historique**: Compare les prédictions avec les valeurs réelles
- **Interface moderne**: Thème technologique avec animations et graphiques interactifs

## 📋 Prérequis

- Python 3.8 ou supérieur
- Les dépendances listées dans `requirements.txt`

## 🔧 Installation

1. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

2. **Entraîner les modèles** (première fois uniquement):
```bash
python train_models.py
```

Cette étape va:
- Charger et préparer les données
- Entraîner tous les modèles
- Sauvegarder les modèles dans le dossier `models/`

**Note**: L'entraînement peut prendre plusieurs minutes selon votre machine.

3. **Lancer l'application web**:
```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 📖 Utilisation

### Prédiction en Temps Réel

1. Sélectionnez un modèle dans la sidebar
2. Choisissez "📈 Prédiction en temps réel"
3. Cliquez sur "🔄 Générer Prédiction"
4. Visualisez la prédiction avec les graphiques interactifs

### Prédiction Historique

1. Sélectionnez un modèle dans la sidebar
2. Choisissez "📅 Prédiction avec données historiques"
3. Sélectionnez une date et une heure
4. Cliquez sur "🔮 Générer Prédiction"
5. Comparez la prédiction avec la valeur réelle

## 📁 Structure du Projet

```
.
├── app.py                              # Application Streamlit principale
├── train_models.py                     # Script d'entraînement des modèles
├── requirements.txt                    # Dépendances Python
├── electricityConsumptionAndProductioction.csv  # Données
├── project.ipynb                       # Notebook original
└── models/                            # Dossier des modèles sauvegardés
    ├── scaler.pkl                     # Scaler pour normalisation
    ├── params.pkl                     # Paramètres du modèle
    ├── decision_tree.pkl              # Modèle Decision Tree
    ├── mlp_model.h5                   # Modèle MLP
    ├── cnn_model.h5                   # Modèle CNN
    ├── lstm_uni_model.h5              # Modèle LSTM Univariate
    └── lstm_multi_model.h5            # Modèle LSTM Multivariate
```

## 🎨 Thème

L'interface utilise un thème technologique moderne avec:
- Fond dégradé sombre (bleu foncé)
- Accents néon (cyan, vert, rose)
- Graphiques interactifs avec Plotly
- Animations et effets visuels

## ⚙️ Configuration

Les paramètres du modèle peuvent être modifiés dans `train_models.py`:
- `window_size`: Taille de la fenêtre temporelle (défaut: 24 heures)
- `train_ratio`: Proportion des données d'entraînement (défaut: 0.8)
- Architecture des modèles (couches, neurones, etc.)

## 📊 Modèles Disponibles

| Modèle | Type | Description |
|--------|------|-------------|
| Decision Tree | Machine Learning | Arbre de décision avec profondeur max 10 |
| MLP | Deep Learning | Réseau de neurones multicouches avec régularisation |
| CNN | Deep Learning | Réseau de neurones convolutifs 1D |
| LSTM (Univariate) | Deep Learning | LSTM bidirectionnel avec seulement la consommation |
| LSTM (Multivariate) | Deep Learning | LSTM bidirectionnel avec toutes les features |

## 🔄 Mise à Jour des Modèles

Pour ré-entraîner les modèles avec de nouvelles données:

1. Remplacez le fichier CSV avec vos nouvelles données
2. Exécutez `python train_models.py`
3. Les nouveaux modèles seront sauvegardés automatiquement

## 🐛 Dépannage

**Erreur: "Impossible de charger les modèles"**
- Assurez-vous d'avoir exécuté `train_models.py` au moins une fois
- Vérifiez que le dossier `models/` contient tous les fichiers nécessaires

**Erreur: "Module not found"**
- Installez toutes les dépendances: `pip install -r requirements.txt`

**L'application est lente**
- L'entraînement initial prend du temps, mais les prédictions sont rapides
- Utilisez un GPU si disponible pour accélérer l'entraînement

## 📝 Notes

- Les modèles sont pré-entraînés pour des performances optimales
- La prédiction en temps réel utilise les 24 dernières heures disponibles
- Tous les modèles utilisent la normalisation MinMax pour de meilleures performances

## 👨‍💻 Développement

Pour contribuer ou modifier l'application:
- Modifiez `app.py` pour changer l'interface
- Modifiez `train_models.py` pour ajuster les modèles
- Le thème CSS peut être personnalisé dans `app.py`

## 📄 Licence

Ce projet est fourni tel quel pour usage éducatif et de démonstration.

