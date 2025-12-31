@echo off
echo ========================================
echo   AI Energy Forecast System
echo ========================================
echo.
echo Activation de l'environnement tf_clean...
call conda activate tf_clean
if errorlevel 1 (
    echo.
    echo ERREUR: Impossible d'activer l'environnement tf_clean
    echo.
    echo Veuillez:
    echo   1. Ouvrir Anaconda Prompt
    echo   2. Taper: conda activate tf_clean
    echo   3. Naviguer ici: cd C:\Users\asus\Desktop\cur
    echo   4. Executer: streamlit run app.py
    echo.
    pause
    exit /b 1
)
echo.
echo Environnement active avec succes!
echo.
echo Verification des modeles...
if not exist "models\scaler.pkl" (
    echo Les modeles ne sont pas entraines.
    echo Voulez-vous les entrainer maintenant? (O/N)
    set /p response=
    if /i "%response%"=="O" (
        echo Entrainement en cours...
        python train_models.py
        echo.
    ) else (
        echo Veuillez d'abord executer: python train_models.py
        pause
        exit /b 1
    )
)
echo.
echo Demarrage de l'application web...
streamlit run app.py
pause

