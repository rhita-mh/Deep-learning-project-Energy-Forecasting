#!/bin/bash

echo "========================================"
echo "  AI Energy Forecast System"
echo "========================================"
echo ""
echo "Vérification des modèles..."
if [ ! -f "models/scaler.pkl" ]; then
    echo "Les modèles ne sont pas entraînés."
    echo "Entraînement des modèles en cours..."
    python train_models.py
    echo ""
fi
echo ""
echo "Démarrage de l'application web..."
streamlit run app.py

