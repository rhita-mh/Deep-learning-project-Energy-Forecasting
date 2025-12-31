@echo off
echo ========================================
echo   Entrainement des Modeles
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
    echo   4. Executer: python train_models.py
    echo.
    pause
    exit /b 1
)
echo.
echo Environnement active avec succes!
echo.
echo Verification de TensorFlow...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>nul
if errorlevel 1 (
    echo ERREUR: TensorFlow n'est pas installe dans tf_clean
    echo Installation en cours...
    pip install tensorflow
)
echo.
echo Demarrage de l'entrainement...
python train_models.py
echo.
pause

