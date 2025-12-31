@echo off
echo ========================================
echo   AI Energy Forecast System
echo ========================================
echo.
echo IMPORTANT: Activez d'abord l'environnement tf_clean !
echo.
echo Si vous utilisez Anaconda:
echo   1. Ouvrez Anaconda Prompt
echo   2. Tapez: conda activate tf_clean
echo   3. Naviguez ici: cd C:\Users\asus\Desktop\cur
echo   4. Puis executez ce script
echo.
pause
echo.
echo Verification des modeles...
if not exist "models\scaler.pkl" (
    echo Les modeles ne sont pas entraines.
    echo Entrainement des modeles en cours...
    python train_models.py
    echo.
)
echo.
echo Demarrage de l'application web...
streamlit run app.py
pause

