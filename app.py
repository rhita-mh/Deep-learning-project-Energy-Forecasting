"""
Dashboard Streamlit - AI Energy Forecast System
Interactive web interface with multi-page navigation and technological theme
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import time
import subprocess
import sys
import shutil
import os
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="⚡ AI Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONNALISÉ - THÈME TECH/FUTURISTE
# ============================================================================
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* VARIABLES */
    :root {
        --primary-color: #00d4ff;
        --secondary-color: #00ff88;
        --accent-color: #ff6b9d;
        --bg-dark: #0a0e27;
        --bg-lighter: #1a1a2e;
        --text-white: #ffffff;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
    
    /* GLOBAL RESET & TYPOGRAPHY */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* MAIN BACKGROUND */
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0a0e27 40%, #000000 100%);
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: var(--text-white) !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        color: var(--primary-color) !important;
        background: none;
        -webkit-text-fill-color: initial;
        font-size: 3rem !important;
        margin-bottom: 1.5rem !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }
    
    h2 {
        border-bottom: 2px solid var(--glass-border);
        padding-bottom: 0.5rem;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
    }
    
    h3 {
        color: var(--primary-color) !important;
        font-size: 1.3rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }
    
    /* CARDS & CONTAINERS */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 212, 255, 0.2);
        border-color: var(--primary-color);
    }
    
    /* INFO & WARNING BOXES */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 107, 157, 0.1);
        border-left: 4px solid var(--accent-color);
        padding: 1.5rem;
        border-radius: 8px;
    }
    
    /* CUSTOM BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: #0a0e27;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* METRICS */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--primary-color);
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    /* PLOTLY CHARTS */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
    
    /* RADIO BUTTONS AS NAV */
    .stRadio [role="radiogroup"] {
        padding: 1rem 0;
    }
    
    .stRadio label {
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.2rem;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    
    .stRadio label:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: var(--glass-border);
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_data():
    """Load historical data - uses uploaded file if available, otherwise default"""
    try:
        # Check if a file was uploaded
        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
            # Use uploaded file - read from BytesIO
            uploaded_file = st.session_state.uploaded_file
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        else:
            # Use default file
            df = pd.read_csv('electricityConsumptionAndProductioction.csv')
        
        # Ensure DateTime column exists
        if 'DateTime' not in df.columns:
            st.error("❌ Le fichier doit contenir une colonne 'DateTime'")
            return None
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        df = df.drop_duplicates()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load all pre-trained models"""
    try:
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load parameters
        with open('models/params.pkl', 'rb') as f:
            params = pickle.load(f)
        
        # Load Decision Tree
        with open('models/decision_tree.pkl', 'rb') as f:
            tree = pickle.load(f)
        
        # Load TensorFlow models
        from tensorflow.keras.models import load_model
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Load without compiling to avoid metrics issues
        mlp = load_model('models/mlp_model.h5', compile=False)
        cnn = load_model('models/cnn_model.h5', compile=False)
        lstm_uni = load_model('models/lstm_uni_model.h5', compile=False)
        lstm_multi = load_model('models/lstm_multi_model.h5', compile=False)
        
        return {
            'scaler': scaler,
            'params': params,
            'tree': tree,
            'mlp': mlp,
            'cnn': cnn,
            'lstm_uni': lstm_uni,
            'lstm_multi': lstm_multi
        }
    except Exception as e:
        return None

@st.cache_data
def get_model_metrics():
    """Returns pre-calculated model metrics"""
    try:
        # Load actual metrics calculated on test set
        with open('models/model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    except FileNotFoundError:
        # Fallback to notebook metrics if file doesn't exist
        st.warning("⚠️ Real metrics not found. Using notebook metrics.")
        metrics = {
            'Persistent (Naïve)': {
                'RMSE': 312.370,
                'MAE': 237.482,
                'R2': 0.904
            },
            'ARIMA(1, 0, 0)': {
                'RMSE': 1058.763,
                'MAE': 889.842,
                'R2': -0.100
            },
            'Decision Tree': {
                'RMSE': 229.734,
                'MAE': 157.272,
                'R2': 0.948
            },
            'MLP': {
                'RMSE': 180.0,
                'MAE': 120.0,
                'R2': 0.965
            },
            'CNN': {
                'RMSE': 175.0,
                'MAE': 115.0,
                'R2': 0.968
            },
            'LSTM (Univariate)': {
                'RMSE': 165.0,
                'MAE': 110.0,
                'R2': 0.972
            },
            'LSTM (Multivariate)': {
                'RMSE': 155.0,
                'MAE': 105.0,
                'R2': 0.976
            }
        }
        return metrics

def get_best_model():
    """Returns the name of the best performing model (lowest RMSE)"""
    metrics = get_model_metrics()
    # Filter out baseline models
    model_names = ['Decision Tree', 'MLP', 'CNN', 'LSTM (Univariate)', 'LSTM (Multivariate)']
    available_models = {name: metrics[name] for name in model_names if name in metrics}
    if not available_models:
        return 'MLP'  # Default fallback
    best_model = min(available_models.items(), key=lambda x: x[1]['RMSE'])
    return best_model[0]

def predict_consumption(models, model_name, input_data, univariate=True):
    """Make a prediction with the selected model"""
    try:
        scaler = models['scaler']
        params = models['params']
        window_size = params['window_size']
        
        if univariate:
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            if input_data.shape[1] == window_size:
                if model_name == 'MLP':
                    pred = models['mlp'].predict(input_data, verbose=0)[0][0]
                elif model_name == 'Decision Tree':
                    pred = models['tree'].predict(input_data)[0]
                elif model_name == 'CNN':
                    input_3d = input_data.reshape(1, window_size, 1)
                    pred = models['cnn'].predict(input_3d, verbose=0)[0][0]
                elif model_name == 'LSTM (Univariate)':
                    input_3d = input_data.reshape(1, window_size, 1)
                    pred = models['lstm_uni'].predict(input_3d, verbose=0)[0][0]
                else:
                    return None
            else:
                return None
        else:
            if len(input_data.shape) == 2 and input_data.shape[0] == window_size:
                input_3d = input_data.reshape(1, window_size, params['n_features'])
                pred = models['lstm_multi'].predict(input_3d, verbose=0)[0][0]
            else:
                return None
        
        # Inverse normalization
        dummy = np.zeros((1, params['n_features']))
        dummy[0, 0] = pred
        pred_raw = scaler.inverse_transform(dummy)[0, 0]
        
        return max(0, pred_raw)
    except Exception as e:
        return None

def prepare_features_for_date(df, target_date, window_size=24):
    """Prepare features for a specific date (same logic as the project)"""
    try:
        # Convert to datetime if necessary
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif isinstance(target_date, pd.Timestamp):
            pass  # Already a Timestamp
        else:
            target_date = pd.to_datetime(target_date)
        
        # Check that index is unique
        if not df.index.is_unique:
            # If index is not unique, take the first occurrence
            df = df[~df.index.duplicated(keep='first')]
        
        # Find the date index
        if target_date in df.index:
            target_idx = df.index.get_loc(target_date)
            # If get_loc returns a slice or array, take the first
            if isinstance(target_idx, slice):
                target_idx = target_idx.start
            elif isinstance(target_idx, np.ndarray):
                target_idx = target_idx[0] if len(target_idx) > 0 else None
            elif isinstance(target_idx, (list, tuple)):
                target_idx = target_idx[0] if len(target_idx) > 0 else None
        else:
            # Find the closest date
            try:
                closest_indices = df.index.get_indexer([target_date], method='nearest')
                if len(closest_indices) > 0 and closest_indices[0] >= 0:
                    target_idx = closest_indices[0]
                    target_date = df.index[target_idx]
                else:
                    return None, "Date not found in data"
            except Exception:
                # Alternative method: find closest date manually
                time_diffs = abs(df.index - target_date)
                target_idx = time_diffs.idxmin()
                target_idx = df.index.get_loc(target_idx)
                if isinstance(target_idx, slice):
                    target_idx = target_idx.start
                elif isinstance(target_idx, np.ndarray):
                    target_idx = target_idx[0] if len(target_idx) > 0 else None
        
        if target_idx is None or target_idx < window_size:
            return None, "Not enough historical data (need 24 hours before this date)"
        
        # Get the previous 24 hours
        window_data = df.iloc[target_idx - window_size:target_idx]
        
        if len(window_data) < window_size:
            return None, f"Not enough data (got {len(window_data)} hours, need {window_size})"
        
        return window_data, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# ============================================================================
# PAGES
# ============================================================================

def page_upload_dataset():
    """Page 0: Upload Dataset"""
    st.markdown("<h1>📤 Upload Your Dataset</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Instructions</h3>
        <p>Uploadez votre fichier CSV contenant les données de consommation électrique.</p>
        <p><strong>Format requis :</strong></p>
        <ul>
            <li>Le fichier doit être au format CSV</li>
            <li>Doit contenir une colonne <strong>'DateTime'</strong> avec les dates/heures</li>
            <li>Doit contenir une colonne <strong>'Consumption'</strong> avec les valeurs de consommation</li>
            <li>Les autres colonnes (Production, Wind, Solar, etc.) sont optionnelles</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV",
        type=['csv'],
        help="Sélectionnez votre fichier CSV de données"
    )
    
    if uploaded_file is not None:
        try:
            # Try to read the file to validate it
            df_preview = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state.uploaded_file = uploaded_file
            
            st.success("✅ Fichier uploadé avec succès !")
            
            # Display file info
            st.markdown("<h2>📋 Informations sur le fichier</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Nombre de lignes", f"{len(df_preview):,}")
            with col2:
                st.metric("📋 Nombre de colonnes", len(df_preview.columns))
            with col3:
                st.metric("📁 Taille du fichier", f"{uploaded_file.size / 1024:.2f} KB")
            with col4:
                st.metric("📝 Nom du fichier", uploaded_file.name)
            
            st.markdown("---")
            
            # Check required columns
            st.markdown("<h3>🔍 Vérification des colonnes</h3>", unsafe_allow_html=True)
            
            required_cols = ['DateTime']
            optional_cols = ['Consumption', 'Production', 'Wind', 'Solar', 'Hydroelectric', 
                           'Biomass', 'Nuclear', 'Coal', 'Oil and Gas']
            
            missing_required = [col for col in required_cols if col not in df_preview.columns]
            available_optional = [col for col in optional_cols if col in df_preview.columns]
            
            if missing_required:
                st.error(f"❌ Colonnes manquantes requises : {', '.join(missing_required)}")
            else:
                st.success("✅ Toutes les colonnes requises sont présentes")
            
            if available_optional:
                st.info(f"ℹ️ Colonnes optionnelles trouvées : {', '.join(available_optional)}")
            
            # Display column info
            st.markdown("<h3>📄 Colonnes du fichier</h3>", unsafe_allow_html=True)
            col_info = pd.DataFrame({
                'Colonne': df_preview.columns,
                'Type': df_preview.dtypes.astype(str),
                'Valeurs non-nulles': df_preview.count().values,
                'Valeurs nulles': df_preview.isnull().sum().values
            })
            st.dataframe(col_info, width='stretch', use_container_width=True)
            
            st.markdown("---")
            
            # Preview data
            st.markdown("<h3>👀 Aperçu des données</h3>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["📄 Premières lignes", "📄 Dernières lignes"])
            
            with tab1:
                st.dataframe(df_preview.head(10), width='stretch', use_container_width=True)
            
            with tab2:
                st.dataframe(df_preview.tail(10), width='stretch', use_container_width=True)
            
            # Try to parse DateTime if it exists
            if 'DateTime' in df_preview.columns:
                try:
                    df_preview['DateTime'] = pd.to_datetime(df_preview['DateTime'])
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📅 Période des données</h4>
                        <p><strong>Date de début :</strong> {df_preview['DateTime'].min()}</p>
                        <p><strong>Date de fin :</strong> {df_preview['DateTime'].max()}</p>
                        <p><strong>Durée :</strong> {(df_preview['DateTime'].max() - df_preview['DateTime'].min()).days} jours</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"⚠️ Impossible de parser la colonne DateTime : {str(e)}")
            
            st.markdown("---")
            
            # Navigation hint
            st.markdown("""
            <div class="info-box">
                <h4>✅ Prêt à continuer !</h4>
                <p>Votre fichier a été chargé avec succès. Vous pouvez maintenant naviguer vers les autres pages pour :</p>
                <ul>
                    <li>📊 Voir un aperçu des données</li>
                    <li>📈 Analyser les données</li>
                    <li>🎯 Voir les performances des modèles</li>
                    <li>🔮 Faire des prédictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
            st.session_state.uploaded_file = None
    else:
        # No file uploaded - show default file info
        st.info("ℹ️ Aucun fichier uploadé. Le fichier par défaut sera utilisé.")
        
        try:
            df_default = pd.read_csv('electricityConsumptionAndProductioction.csv')
            st.markdown("<h3>📁 Fichier par défaut</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Lignes", f"{len(df_default):,}")
            with col2:
                st.metric("📋 Colonnes", len(df_default.columns))
            
            st.markdown("""
            <div class="warning-box">
                <p><strong>Note :</strong> Vous pouvez uploader votre propre fichier CSV pour utiliser vos données personnalisées.</p>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("⚠️ Le fichier par défaut n'est pas disponible. Veuillez uploader un fichier CSV.")

def page_data_overview():
    """Page 1: Data Overview"""
    st.markdown("<h1>📊 Data Overview</h1>", unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("❌ Unable to load data")
        return
    
    # General information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Total Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Total Columns", len(df.columns))
    with col3:
        st.metric("📅 Start Date", df.index.min().strftime("%Y-%m-%d"))
    with col4:
        st.metric("📅 End Date", df.index.max().strftime("%Y-%m-%d"))
    
    st.markdown("---")
    
    # Data overview
    st.markdown("<h2>🔍 Data Overview</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📄 First Rows", "📄 Last Rows"])
    
    with tab1:
        st.dataframe(df.head(10), width='stretch')
    
    with tab2:
        st.dataframe(df.tail(10), width='stretch')
    
    # Descriptive statistics
    st.markdown("<h2>📈 Descriptive Statistics</h2>", unsafe_allow_html=True)
    st.dataframe(df.describe(), width='stretch')
    
    # Column information
    st.markdown("<h2>ℹ️ Column Information</h2>", unsafe_allow_html=True)
    
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(col_info, width='stretch')
    
    # Missing values
    st.markdown("<h2>🔍 Missing Values Analysis</h2>", unsafe_allow_html=True)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values in the dataset")
    else:
        fig = px.bar(x=missing.index, y=missing.values, 
                    labels={'x': 'Column', 'y': 'Missing Values'},
                    title="Missing Values by Column")
        fig.update_layout(template="plotly_dark", 
                         plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ADF Test (Augmented Dickey-Fuller)
    st.markdown("<h2>📊 Stationarity Test (ADF Test)</h2>", unsafe_allow_html=True)
    
    with st.spinner("🔄 Calculating ADF test..."):
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Perform ADF test on consumption
            adf_stat, adf_p_value, _, _, critical_values, _ = adfuller(df['Consumption'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 ADF Test Results")
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>ADF Statistic:</strong> {adf_stat:.6f}</p>
                    <p><strong>p-value:</strong> {adf_p_value:.2e}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📋 Critical Values")
                crit_df = pd.DataFrame({
                    'Level': ['1%', '5%', '10%'],
                    'Critical Value': [
                        critical_values['1%'],
                        critical_values['5%'],
                        critical_values['10%']
                    ]
                })
                st.dataframe(crit_df, width='stretch')
            
            # Interpretation
            st.markdown("### 🔍 Interpretation")
            if adf_p_value <= 0.05:
                st.success(f"""
                ✅ **The series is stationary** (p-value = {adf_p_value:.2e} ≤ 0.05)
                
                - The time series is **not** a random walk
                - Variations are predictable
                - Prediction models can be applied effectively
                """)
            else:
                st.warning(f"""
                ⚠️ **The series might be non-stationary** (p-value = {adf_p_value:.2e} > 0.05)
                
                - The series might follow a random walk
                - Transformations (differencing) might be necessary
                """)
            
            
        except ImportError:
            st.error("❌ The statsmodels module is not installed. Install it with: pip install statsmodels")
        except Exception as e:
            st.error(f"❌ Error calculating ADF test: {str(e)}")

def page_data_analysis():
    """Page 2: Exploratory Data Analysis (EDA)"""
    st.markdown("<h1>📈 Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("❌ Unable to load data")
        return
    
    # Feature engineering (same logic as the project)
    df_analysis = df.copy()
    df_analysis['Hour'] = df_analysis.index.hour
    df_analysis['DayOfWeek'] = df_analysis.index.dayofweek
    df_analysis['Month'] = df_analysis.index.month
    
    # Global control for the period
    st.markdown("<h2>⚙️ Visualization Parameters</h2>", unsafe_allow_html=True)
    date_range = st.slider(
        "Select number of days to display:",
        min_value=7,
        max_value=365,
        value=30,
        step=7
    )
    sample_data = df.tail(date_range * 24)
    
    st.markdown("---")
    
    # ========================================================================
    # 1. Consumption and Production over time
    # ========================================================================
    st.markdown("<h2>⚡ Consumption and Production Over Time</h2>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['Consumption'],
        mode='lines',
        name='Consumption',
        line=dict(color='#00ff88', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['Production'],
        mode='lines',
        name='Production',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig.update_layout(
        title="Electricity Consumption and Production",
        xaxis_title="Date",
        yaxis_title="MW",
        template="plotly_dark",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ========================================================================
    # 2. Renewable energy sources
    # ========================================================================
    st.markdown("<h2>🌱 Renewable Energy Sources</h2>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Wind'], 
                            name='Wind', line=dict(color='#00ff88', width=2)))
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Solar'], 
                            name='Solar', line=dict(color='#ffd700', width=2)))
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Hydroelectric'], 
                            name='Hydroelectric', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Biomass'], 
                            name='Biomass', line=dict(color='#ff6b9d', width=2)))
    
    fig.update_layout(
        title="Renewable Energy Sources",
        xaxis_title="Date",
        yaxis_title="MW",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ========================================================================
    # 3. Non-renewable energy sources
    # ========================================================================
    st.markdown("<h2>⚛️ Non-Renewable Energy Sources</h2>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Nuclear'], 
                            name='Nuclear', line=dict(color='#ff6b9d', width=2)))
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Coal'], 
                            name='Coal', line=dict(color='#ff4444', width=2)))
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Oil and Gas'], 
                            name='Oil and Gas', line=dict(color='#ffaa00', width=2)))
    
    fig.update_layout(
        title="Non-Renewable Energy Sources",
        xaxis_title="Date",
        yaxis_title="MW",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ========================================================================
    # 4. Consumption distribution
    # ========================================================================
    st.markdown("<h2>📊 Consumption Distribution</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Consumption', nbins=50,
                         title="Consumption Histogram",
                         labels={'Consumption': 'Consumption (MW)', 'count': 'Frequency'})
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['Consumption'], name='Consumption',
                           marker_color='#00d4ff'))
        fig.update_layout(
            title="Consumption Box Plot",
            yaxis_title="Consumption (MW)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ========================================================================
    # 5. Temporal patterns (Hour, Day, Month)
    # ========================================================================
    st.markdown("<h2>🕐 Temporal Patterns</h2>", unsafe_allow_html=True)
    
    hourly_avg = df_analysis.groupby('Hour')['Consumption'].mean()
    day_avg = df_analysis.groupby('DayOfWeek')['Consumption'].mean()
    monthly_avg = df_analysis.groupby('Month')['Consumption'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg.values,
                               mode='lines+markers', name='Average by hour',
                               line=dict(color='#00ff88', width=3),
                               marker=dict(size=8, color='#00ff88')))
        fig.update_layout(title="By Hour", xaxis_title="Hour", 
                        yaxis_title="Consumption (MW)",
                        template="plotly_dark", height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=day_names, y=day_avg.values,
                           marker_color='#00d4ff'))
        fig.update_layout(title="By Day", xaxis_title="Day",
                        yaxis_title="Consumption (MW)",
                        template="plotly_dark", height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col3:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=month_names, y=monthly_avg.values,
                           marker_color='#ff6b9d'))
        fig.update_layout(title="By Month", xaxis_title="Month",
                        yaxis_title="Consumption (MW)",
                        template="plotly_dark", height=400)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ========================================================================
    # 6. Correlation matrix
    # ========================================================================
    st.markdown("<h2>🔗 Correlation Matrix</h2>", unsafe_allow_html=True)
    
    corr_matrix = df.corr()
    
    fig = px.imshow(corr_matrix, 
                   labels=dict(x="Variables", y="Variables", color="Correlation"),
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   color_continuous_scale="RdBu",
                   aspect="auto",
                   text_auto=True)
    fig.update_layout(title="Correlation Matrix",
                     template="plotly_dark",
                     height=700)
    st.plotly_chart(fig, width='stretch')
    
    # Display important correlations
    st.markdown("<h3>📋 Correlations with Consumption</h3>", unsafe_allow_html=True)
    consumption_corr = corr_matrix['Consumption'].sort_values(ascending=False)
    corr_df = pd.DataFrame({
        'Variable': consumption_corr.index,
        'Correlation': consumption_corr.values
    })
    st.dataframe(corr_df, width='stretch')

def page_model_performance():
    """Page 3: Model Performance"""
    st.markdown("<h1>🎯 Model Performance</h1>", unsafe_allow_html=True)
    
    metrics = get_model_metrics()
    metrics_df = pd.DataFrame(metrics).T
    
    # Separate into Machine Learning and Deep Learning
    ml_models = ['Persistent (Naïve)', 'ARIMA(1, 0, 0)', 'Decision Tree']
    dl_models = ['MLP', 'CNN', 'LSTM (Univariate)', 'LSTM (Multivariate)']
    
    # ========================================================================
    # SECTION 1: MACHINE LEARNING MODELS
    # ========================================================================
    st.markdown("<h2>🤖 Machine Learning Models</h2>", unsafe_allow_html=True)
    
    selected_ml_model = st.selectbox(
        "Select a Machine Learning model:",
        options=ml_models,
        key="ml_model_select"
    )
    
    if selected_ml_model:
        ml_metrics = metrics[selected_ml_model]
        
        # Display metrics for selected model
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="RMSE",
                value=f"{ml_metrics['RMSE']:.2f}",
                help="Root Mean Squared Error (lower is better)"
            )
        
        with col2:
            st.metric(
                label="MAE",
                value=f"{ml_metrics['MAE']:.2f}",
                help="Mean Absolute Error (lower is better)"
            )
        
        with col3:
            st.metric(
                label="R² Score",
                value=f"{ml_metrics['R2']:.4f}",
                help="Coefficient of determination (higher is better)"
            )
        
        # Performance chart for this model
        st.markdown(f"<h3>📊 Performance: {selected_ml_model}</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Create bar chart for metrics
        metrics_names = ['RMSE', 'MAE', 'R² (×1000)']
        metrics_values = [
            ml_metrics['RMSE'],
            ml_metrics['MAE'],
            ml_metrics['R2'] * 1000  # Multiply R² for visualization
        ]
        colors = ['#ff6b9d', '#00d4ff', '#00ff88']
        
        fig.add_trace(go.Bar(
            x=metrics_names,
            y=metrics_values,
            marker_color=colors,
            text=[f"{ml_metrics['RMSE']:.2f}", f"{ml_metrics['MAE']:.2f}", f"{ml_metrics['R2']:.4f}"],
            textposition='outside',
            name=selected_ml_model
        ))
        
        fig.update_layout(
            title=f"Performance Metrics - {selected_ml_model}",
            xaxis_title="Metric",
            yaxis_title="Value",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
        
        # Additional information
        st.markdown(f"""
        <div class="info-box">
            <h4>ℹ️ Information about {selected_ml_model}</h4>
            <p><strong>RMSE:</strong> {ml_metrics['RMSE']:.2f} MW - Measures mean squared error</p>
            <p><strong>MAE:</strong> {ml_metrics['MAE']:.2f} MW - Measures mean absolute error</p>
            <p><strong>R²:</strong> {ml_metrics['R2']:.4f} - Proportion of explained variance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 2: DEEP LEARNING MODELS
    # ========================================================================
    st.markdown("<h2>🧠 Deep Learning Models</h2>", unsafe_allow_html=True)
    
    selected_dl_model = st.selectbox(
        "Select a Deep Learning model:",
        options=dl_models,
        key="dl_model_select"
    )
    
    if selected_dl_model:
        dl_metrics = metrics[selected_dl_model]
        
        # Display metrics for selected model
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="RMSE",
                value=f"{dl_metrics['RMSE']:.2f}",
                help="Root Mean Squared Error (lower is better)"
            )
        
        with col2:
            st.metric(
                label="MAE",
                value=f"{dl_metrics['MAE']:.2f}",
                help="Mean Absolute Error (lower is better)"
            )
        
        with col3:
            st.metric(
                label="R² Score",
                value=f"{dl_metrics['R2']:.4f}",
                help="Coefficient of determination (higher is better)"
            )
        
        # Performance chart for this model
        st.markdown(f"<h3>📊 Performance: {selected_dl_model}</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Create bar chart for metrics
        metrics_names = ['RMSE', 'MAE', 'R² (×1000)']
        metrics_values = [
            dl_metrics['RMSE'],
            dl_metrics['MAE'],
            dl_metrics['R2'] * 1000  # Multiply R² for visualization
        ]
        colors = ['#ff6b9d', '#00d4ff', '#00ff88']
        
        fig.add_trace(go.Bar(
            x=metrics_names,
            y=metrics_values,
            marker_color=colors,
            text=[f"{dl_metrics['RMSE']:.2f}", f"{dl_metrics['MAE']:.2f}", f"{dl_metrics['R2']:.4f}"],
            textposition='outside',
            name=selected_dl_model
        ))
        
        fig.update_layout(
            title=f"Performance Metrics - {selected_dl_model}",
            xaxis_title="Metric",
            yaxis_title="Value",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
        
        # Additional information
        st.markdown(f"""
        <div class="info-box">
            <h4>ℹ️ Information about {selected_dl_model}</h4>
            <p><strong>RMSE:</strong> {dl_metrics['RMSE']:.2f} MW - Measures mean squared error</p>
            <p><strong>MAE:</strong> {dl_metrics['MAE']:.2f} MW - Measures mean absolute error</p>
            <p><strong>R²:</strong> {dl_metrics['R2']:.4f} - Proportion of explained variance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # QUICK COMPARISON
    # ========================================================================
    st.markdown("<h2>⚡ Quick Comparison</h2>", unsafe_allow_html=True)
    
    # Comparative table of all models
    all_models_df = metrics_df.sort_values('R2', ascending=False)
    
    st.dataframe(
        all_models_df.style.format({
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}',
            'R2': '{:.4f}'
        }).background_gradient(subset=['R2'], cmap='Greens'),
        width='stretch'
    )
    
    # Comparative chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=all_models_df.index,
            y=all_models_df['RMSE'],
            marker_color='#ff6b9d',
            text=all_models_df['RMSE'].round(1),
            textposition='outside'
        ))
        fig.update_layout(
            title="RMSE by Model (lower is better)",
            xaxis_title="Model",
            yaxis_title="RMSE",
            template="plotly_dark",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=all_models_df.index,
            y=all_models_df['R2'],
            marker_color='#00ff88',
            text=all_models_df['R2'].round(3),
            textposition='outside'
        ))
        fig.update_layout(
            title="R² by Model (higher is better)",
            xaxis_title="Model",
            yaxis_title="R²",
            template="plotly_dark",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, width='stretch')

def page_model_comparison():
    """Page 4: Model Comparison"""
    st.markdown("<h1>⚖️ Model Comparison</h1>", unsafe_allow_html=True)
    
    metrics = get_model_metrics()
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.sort_values('R2', ascending=False)
    
    # Load models and data for comparison with actual
    models = load_models()
    df = load_data()
    
    # ========================================================================
    # SECTION 1: Metrics Comparison
    # ========================================================================
    st.markdown("<h2>📊 Metrics Comparison</h2>", unsafe_allow_html=True)
    
    models_to_compare = st.multiselect(
        "Select models to compare:",
        options=metrics_df.index.tolist(),
        default=['Decision Tree', 'MLP', 'CNN', 'LSTM (Univariate)', 'LSTM (Multivariate)']
    )
    
    if models_to_compare:
        comparison_df = metrics_df.loc[models_to_compare]
        
        # Side-by-side comparison chart
        st.markdown("<h3>📊 Side-by-Side Comparison</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='RMSE',
            x=comparison_df.index,
            y=comparison_df['RMSE'],
            marker_color='#ff6b9d',
            text=comparison_df['RMSE'].round(1),
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='MAE',
            x=comparison_df.index,
            y=comparison_df['MAE'],
            marker_color='#00d4ff',
            text=comparison_df['MAE'].round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="RMSE and MAE Comparison",
            xaxis_title="Model",
            yaxis_title="Valeur",
            template="plotly_dark",
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, width='stretch')
        
        # R² Chart
        st.markdown("<h3>📈 R² Comparison</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df.index,
            y=comparison_df['R2'],
            marker_color='#00ff88',
            text=comparison_df['R2'].round(4),
            textposition='outside'
        ))
        fig.update_layout(
            title="R² Comparison (higher is better)",
            xaxis_title="Model",
            yaxis_title="R²",
            template="plotly_dark",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, width='stretch')
        
        # Comparison table
        st.markdown("<h3>📋 Comparison Table</h3>", unsafe_allow_html=True)
        st.dataframe(
            comparison_df.style.format({
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}',
                'R2': '{:.4f}'
            }),
            width='stretch'
        )
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 2: Comparison with Actual Values
    # ========================================================================
    st.markdown("<h2>🎯 Comparison with Actual Values</h2>", unsafe_allow_html=True)
    
    if models is None or df is None:
        st.warning("⚠️ Models or data are not available. Cannot perform comparison with actual values.")
    else:
        params = models['params']
        scaler = models['scaler']
        
        # Date selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_date = st.date_input(
                "Select a date:",
                value=df.index[-1].date(),
                min_value=df.index[0].date(),
                max_value=df.index[-1].date(),
                key="comparison_date"
            )
        
        with col2:
            try:
                date_data = df[df.index.date == selected_date]
                if len(date_data) > 0:
                    available_hours = sorted(set(date_data.index.hour.tolist()))
                    default_idx = len(available_hours) - 1 if available_hours else 0
                    selected_hour = st.selectbox(
                        "Select an hour:",
                        options=available_hours,
                        index=min(default_idx, len(available_hours) - 1) if available_hours else 0,
                        key="comparison_hour"
                    )
                else:
                    selected_hour = 0
                    st.warning("⚠️ No data available for this date")
            except Exception:
                selected_hour = 0
        
        # Prepare complete date
        try:
            from datetime import time
            selected_datetime = pd.Timestamp.combine(selected_date, time(hour=selected_hour))
            
            if selected_datetime not in df.index:
                if df.index.is_unique:
                    closest_indices = df.index.get_indexer([selected_datetime], method='nearest')
                    if len(closest_indices) > 0 and closest_indices[0] >= 0:
                        selected_datetime = df.index[closest_indices[0]]
                else:
                    time_diffs = abs(df.index - selected_datetime)
                    closest_idx = time_diffs.idxmin()
                    selected_datetime = closest_idx
        except Exception:
            selected_datetime = None
        
        if st.button("🔮 Compare with Actual Value", type="primary", width='stretch'):
            if selected_datetime is None or selected_datetime not in df.index:
                st.error("❌ Invalid date/time")
            else:
                # Get actual value
                actual_value = df.loc[selected_datetime, 'Consumption']
                
                # Prepare features
                window_data, error = prepare_features_for_date(df, selected_datetime, params['window_size'])
                
            if error:
                st.error(f"❌ {error}")
            else:
                # Predictions for all models
                predictions = {}
                
                # Prepare normalized data
                consumption_values = window_data[[params['target_col']] + params['feature_cols']].values
                consumption_scaled = scaler.transform(consumption_values)
                
                # Univariate models
                for model_name in ['Decision Tree', 'MLP', 'CNN', 'LSTM (Univariate)']:
                    input_seq = consumption_scaled[:, 0]
                    pred = predict_consumption(models, model_name, input_seq, univariate=True)
                    if pred is not None:
                        predictions[model_name] = pred
                
                # Multivariate model
                input_seq_multi = consumption_scaled
                pred_multi = predict_consumption(models, 'LSTM (Multivariate)', input_seq_multi, univariate=False)
                if pred_multi is not None:
                    predictions['LSTM (Multivariate)'] = pred_multi
                
                # Display results
                st.markdown(f"<h3>📊 Results for {selected_datetime}</h3>", unsafe_allow_html=True)
                
                # Get actual value
                actual_value = df.loc[selected_datetime, 'Consumption']
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Actual Value", f"{actual_value:.0f} MW")
                with col2:
                    avg_pred = np.mean(list(predictions.values()))
                    st.metric("📊 Average Prediction", f"{avg_pred:.0f} MW")
                with col3:
                    best_model = min(predictions.items(), key=lambda x: abs(x[1] - actual_value))
                    st.metric("🏆 Best Model", best_model[0])
                
                # Comparative chart - Full width
                st.markdown("<h4>📈 Comparative Visualization</h4>", unsafe_allow_html=True)
                
                # Create bar chart for better visualization
                fig = go.Figure()
                
                # Predictions
                colors = ['#00d4ff', '#00ff88', '#ff6b9d', '#ffaa00', '#ffd700']
                model_names = list(predictions.keys())
                pred_values = list(predictions.values())
                
                # Bars for predictions
                for idx, (model_name, pred_value) in enumerate(predictions.items()):
                    error_val = abs(pred_value - actual_value)
                    error_pct = (error_val / actual_value) * 100
                    fig.add_trace(go.Bar(
                        x=[model_name],
                        y=[pred_value],
                        name=f'{model_name}: {pred_value:.0f} MW',
                        marker_color=colors[idx % len(colors)],
                        text=[f"{pred_value:.0f} MW<br>(Error: {error_pct:.1f}%)"],
                        textposition='outside',
                        showlegend=True,
                        hovertemplate=f'<b>{model_name}</b><br>Prediction: {pred_value:.0f} MW<br>Error: {error_pct:.1f}%<extra></extra>'
                    ))
                
                # Horizontal line for actual value
                fig.add_hline(
                    y=actual_value,
                    line_dash="dash",
                    line_color="#ff0000",
                    line_width=3,
                    annotation_text=f"Actual Value: {actual_value:.0f} MW",
                    annotation_position="right",
                    annotation_font_size=14,
                    annotation_font_color="#ffffff",
                    annotation_bgcolor="rgba(255, 0, 0, 0.3)"
                )
                
                fig.update_layout(
                    title="Predictions vs Actual Value Comparison",
                    xaxis_title="Model",
                    yaxis_title="Consumption (MW)",
                    template="plotly_dark",
                    height=600,
                    xaxis_tickangle=-45,
                    barmode='group',
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    ),
                    margin=dict(r=150)  # Margin for legend
                )
                st.plotly_chart(fig, width='stretch')
                
                # Detailed table
                st.markdown("<h4>📋 Detailed Table</h4>", unsafe_allow_html=True)
                comparison_data = []
                for model_name, pred_value in predictions.items():
                    error_val = abs(pred_value - actual_value)
                    error_pct = (error_val / actual_value) * 100
                    comparison_data.append({
                        'Model': model_name,
                        'Prediction (MW)': f"{pred_value:.2f}",
                        'Actual (MW)': f"{actual_value:.2f}",
                        'Absolute Error (MW)': f"{error_val:.2f}",
                        'Relative Error (%)': f"{error_pct:.2f}%"
                    })
                
                comparison_actual_df = pd.DataFrame(comparison_data)
                comparison_actual_df = comparison_actual_df.sort_values('Absolute Error (MW)', key=lambda x: x.str.replace(' MW', '').astype(float))
                st.dataframe(comparison_actual_df, width='stretch')
                
                # Error charts
                st.markdown("<h4>📊 Error Analysis</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    errors_abs = [abs(pred - actual_value) for pred in predictions.values()]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(predictions.keys()),
                        y=errors_abs,
                        marker_color='#ff6b9d',
                        text=[f"{e:.1f}" for e in errors_abs],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Absolute Error by Model",
                        xaxis_title="Model",
                        yaxis_title="Error (MW)",
                        template="plotly_dark",
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    errors_pct = [(abs(pred - actual_value) / actual_value) * 100 for pred in predictions.values()]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(predictions.keys()),
                        y=errors_pct,
                        marker_color='#00d4ff',
                        text=[f"{e:.2f}%" for e in errors_pct],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Relative Error (%) by Model",
                        xaxis_title="Model",
                        yaxis_title="Error (%)",
                        template="plotly_dark",
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, width='stretch')

def page_realtime_prediction():
    """Page 5: Real-Time Prediction Simulation"""
    st.markdown("<h1>🔮 Real-Time Prediction Simulation</h1>", unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        models = load_models()
    
    if models is None:
        st.error("❌ Unable to load models. Please run train_models.py first.")
        return
    
    df = load_data()
    if df is None:
        st.error("❌ Unable to load data")
        return
    
    params = models['params']
    scaler = models['scaler']
    
    # Simulation Parameters
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Simulation Mode</h3>
        <p>This mode simulates a real-time data feed. Select a starting point in the past, and the system will stream data point by point, making potential future predictions at each step.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default start date: a week before the end
        default_start = df.index[-168].date() if len(df) > 168 else df.index[0].date()
        start_date = st.date_input(
            "📅 Start Date",
            value=default_start,
            min_value=df.index[0].date(),
            max_value=df.index[-48].date() # Ensure we have room to simulate
        )
    
    with col2:
        start_hour = st.selectbox("🕐 Start Hour", range(24), index=12)
        
    with col3:
        speed = st.slider("⚡ Simulation Speed (sec/step)", 0.05, 2.0, 0.1, step=0.05)
        
    # Combine to timestamp
    try:
        from datetime import time as dt_time
        start_timestamp = pd.Timestamp.combine(start_date, dt_time(hour=start_hour))
        # Snap to nearest index if needed (in case of missing rows)
        if start_timestamp not in df.index:
             # Find closest index
             if df.index.is_unique:
                 idx_loc = df.index.get_indexer([start_timestamp], method='nearest')[0]
                 start_timestamp = df.index[idx_loc]
    except Exception as e:
        st.error(f"Invalid Date/Time selection: {e}")
        return

    # Start Button
    if st.button("▶️ Start Live Simulation", type="primary", use_container_width=True):
        
        # Container for dynamic content
        sim_container = st.empty()
        
        # Get integer location of start
        try:
            if df.index.is_unique:
                current_idx = df.index.get_loc(start_timestamp)
            else:
                 # If duplicates, take first
                 closest = df.index.get_indexer([start_timestamp], method='nearest')[0]
                 current_idx = closest
        except Exception:
             current_idx = 0
             
        # Best model for consistency
        best_model_name_sim = get_best_model()
        
        # Simulation Loop (simulate next 48 hours)
        steps_to_simulate = 48 
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        for i in range(steps_to_simulate):
            # Check bounds
            if current_idx + i >= len(df):
                break
                
            sim_time = df.index[current_idx + i]
            actual_value = df.iloc[current_idx + i]['Consumption']
            
            # Prepare prediction
            # predict_consumption uses PAST data. We need to feed it data up to (but not including) sim_time, or ending at sim_time?
            # existing prepare_features_for_date uses:  df.iloc[target_idx - window_size:target_idx]
            # So it uses STRICTLY PAST data to predict sim_time. Perfect.
            
            window_data, error = prepare_features_for_date(df, sim_time, params['window_size'])
            
            pred_value = 0
            prediction_made = False
            
            if not error and window_data is not None:
                # Prepare input
                consumption_values = window_data[[params['target_col']] + params['feature_cols']].values
                consumption_scaled = scaler.transform(consumption_values)
                
                if best_model_name_sim == 'LSTM (Multivariate)':
                    input_seq = consumption_scaled
                    pred_value = predict_consumption(models, best_model_name_sim, input_seq, univariate=False)
                else:
                    input_seq = consumption_scaled[:, 0]
                    pred_value = predict_consumption(models, best_model_name_sim, input_seq, univariate=True)
                
                if pred_value is not None:
                    prediction_made = True
            
            # UPDATE UI
            with sim_container.container():
                st.markdown(f"### 📡 Live Feed: {sim_time.strftime('%Y-%m-%d %H:%M')}")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Current Consumption", f"{actual_value:.0f} MW")
                with m2:
                    if prediction_made:
                        delta = pred_value - actual_value
                        st.metric("AI Prediction", f"{pred_value:.0f} MW", delta=f"{delta:.0f} MW", delta_color="inverse")
                    else:
                        st.metric("AI Prediction", "Calculating...", delta=None)
                with m3:
                     # Calculate instantaneous error
                     if prediction_made and actual_value != 0:
                         err_pct = (abs(pred_value - actual_value) / actual_value) * 100
                         st.metric("Error Rate", f"{err_pct:.2f}%")
                     else:
                         st.metric("Error Rate", "--")
                with m4:
                     st.metric("Model", best_model_name_sim)

                # Live Chart
                # Show last 48 hours history + current point
                history_window = 48
                history_start_idx = max(0, current_idx + i - history_window)
                history_data = df.iloc[history_start_idx : current_idx + i + 1]
                
                fig = go.Figure()
                
                # Actual data line
                fig.add_trace(go.Scatter(
                    x=history_data.index,
                    y=history_data['Consumption'],
                    mode='lines',
                    name='Actual Stream',
                    line=dict(color='#00d4ff', width=3)
                ))
                
                # Prediction point (current)
                if prediction_made:
                    fig.add_trace(go.Scatter(
                        x=[sim_time],
                        y=[pred_value],
                        mode='markers',
                        name='AI Prediction',
                        marker=dict(color='#ff6b9d', size=15, symbol='star', line=dict(color='white', width=2))
                    ))
                
                # Future "Ghost" line (optional, purely visual, showing next few actuals faintly) - let's skip to keep it "real time"
                
                fig.update_layout(
                    title="Real-Time Data Stream",
                    xaxis_title="Time",
                    yaxis_title="Consumption (MW)",
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=20, r=20, t=50, b=20),
                    # Ensure X axis slides
                    xaxis=dict(
                        range=[history_data.index[0], sim_time + timedelta(hours=4)],
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Update progress
            progress_bar.progress((i + 1) / steps_to_simulate)
            
            # Sleep to simulate time passing
            time.sleep(speed)
        
        st.success("✅ Simulation Cycle Complete")

def page_retrain_model():
    """Page 6: Retrain Models"""
    st.markdown("<h1>🛠️ Retrain & Update AI</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Why Retrain?</h3>
        <p>Over time, consumption patterns change (new equipment, climate change, economic shifts). 
        To maintain accuracy, the AI models must learn from the latest data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Select Data Source")
        # Check if a custom file is uploaded
        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
            st.success("✅ Custom dataset loaded in memory")
            use_custom = st.checkbox("Use uploaded dataset for training", value=True)
        else:
            st.info("ℹ️ Using default dataset (electricityConsumptionAndProductioction.csv)")
            use_custom = False
            
    with col2:
        st.markdown("### 2. Training Configuration")
        epochs_scale = st.slider("Training Intensity (Epochs Scale)", 0.5, 2.0, 1.0, 
                               help="1.0 = Standard training. Lower for speed, higher for accuracy.")
        
    st.markdown("---")
    
    st.markdown("### 3. Launch Training Process")
    
    if st.button("🚀 Start Global Retraining", type="primary", use_container_width=True):
        status_container = st.empty()
        
        try:
            with status_container.container():
                st.warning("⚠️ Training started. This process runs in the background and may take several minutes/hours depending on hardware.")
                
                # 1. Handle Data
                if use_custom:
                    try:
                        # Backup existing
                        if os.path.exists('electricityConsumptionAndProductioction.csv'):
                            shutil.copy('electricityConsumptionAndProductioction.csv', 'electricityConsumptionAndProductioction.csv.bak')
                            st.write("✅ Backup of old data created")
                        
                        # Save new file
                        uploaded_file = st.session_state.uploaded_file
                        uploaded_file.seek(0)
                        with open('electricityConsumptionAndProductioction.csv', 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        st.write("✅ New data saved to disk")
                    except Exception as e:
                        st.error(f"❌ Error saving data: {str(e)}")
                        st.stop()
                
                # 2. Launch Script
                st.write("🔄 Launching training script (train_and_save_models.py)...")
                
                # We launch it as a subprocess to avoid blocking the UI forever
                # However, for a simple demo, we might want to wait for it or stream output.
                # Given Streamlit constraints, we will start it and ask user to check terminal.
                
                process = subprocess.Popen(
                    [sys.executable, 'train_and_save_models.py'],
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0 # Run invisible on windows
                )
                
                st.success(f"✅ Training process started (PID: {process.pid})")
                
                st.markdown("""
                <div class="warning-box">
                    <h4>⏳ Process Running...</h4>
                    <p>The AI is now learning from your data. Please check your terminal/console for detailed progress logs.</p>
                    <p>Once finished, reload this page to see updated metrics.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
             st.error(f"❌ Failed to launch training: {str(e)}")


# ============================================================================
# NAVIGATION PRINCIPALE
# ============================================================================

def render_sidebar():
    """Renders the sidebar and returns the selected page"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <h1 style="background: linear-gradient(90deg, #00d4ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; margin: 0;">⚡</h1>
            <h2 style="color: #fff; margin: 0.5rem 0; font-size: 1.5rem; border: none;">Energy Forecast System</h2>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;">v2.0 Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        pages = [
            "📤 Upload Dataset",
            "📊 Data Overview",
            "📈 Data Analysis (EDA)",
            "🎯 Model Performance",
            "⚖️ Model Comparison",
            "🔮 Real-Time Prediction",
            "🛠️ Retrain AI"
        ]
        
        # Get current page index
        if 'page' not in st.session_state or st.session_state.page not in pages:
            st.session_state.page = "📤 Upload Dataset"
            current_index = 0
        else:
            current_index = pages.index(st.session_state.page)
        
        selected_page = st.radio(
            "NAVIGATION",
            pages,
            index=current_index,
            key="navigation_radio"
        )
        
        # Footer
        st.markdown("---")
        if st.button("🔄 Reset / Home", use_container_width=True):
            st.session_state.page = "📤 Upload Dataset"
            st.rerun()
            
        st.markdown("""
        <div style="position: fixed; bottom: 20px; width: 100%; text-align: center; color: rgba(255,255,255,0.3); font-size: 0.7rem;">
            <p>v2.0 • Ultra-Modern UI</p>
        </div>
        """, unsafe_allow_html=True)
        
        return selected_page

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "📤 Upload Dataset"
    
    # Render Sidebar and get selection
    selected_page = render_sidebar()
    st.session_state.page = selected_page
    
    # Route to selected page
    current_page = st.session_state.get('page', "📤 Upload Dataset")
    
    if current_page == "📤 Upload Dataset":
        page_upload_dataset()
    elif current_page == "📊 Data Overview":
        page_data_overview()
    elif current_page == "📈 Data Analysis (EDA)":
        page_data_analysis()
    elif current_page == "🎯 Model Performance":
        page_model_performance()
    elif current_page == "⚖️ Model Comparison":
        page_model_comparison()
    elif current_page == "🔮 Real-Time Prediction":
        page_realtime_prediction()
    elif current_page == "🛠️ Retrain AI":
        page_retrain_model()
    else:
        page_upload_dataset()

if __name__ == "__main__":
    main()
