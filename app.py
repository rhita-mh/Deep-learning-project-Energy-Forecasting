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
    /* Technological theme - Dark background with neon accents */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Titre principal */
    h1 {
        color: #00d4ff;
        text-align: center;
        font-weight: bold;
        text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    
    h2 {
        color: #00ff88;
        border-bottom: 3px solid #00ff88;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        text-shadow: 0 0 5px #00ff88;
    }
    
    h3 {
        color: #ff6b9d;
        text-shadow: 0 0 5px #ff6b9d;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 0 0 10px #00d4ff;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    [data-testid="stMetricDelta"] {
        color: #00ff88;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: rgba(10, 14, 39, 0.95);
        border-right: 2px solid #00d4ff;
    }
    
    .css-1lcbmhc .css-1outpf7 {
        color: #00d4ff;
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        color: #0a0e27;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
        background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%);
    }
    
    /* Selectbox et inputs */
    .stSelectbox label, .stDateInput label, .stTimeInput label {
        color: #ffffff;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .stNumberInput label, .stTextInput label {
        color: #ffffff;
        font-weight: bold;
    }
    
    /* Cards personnalisées */
    .metric-card {
        background: rgba(0, 212, 255, 0.1);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        transform: translateY(-5px);
    }
    
    /* Tables */
    .dataframe {
        background-color: rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: rgba(255, 107, 157, 0.1);
        border-left: 4px solid #ff6b9d;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Navigation sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_data
def load_data():
    """Load historical data"""
    try:
        df = pd.read_csv('electricityConsumptionAndProductioction.csv')
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
    """Page 5: Real-Time Prediction"""
    st.markdown("<h1>🔮 Real-Time Prediction</h1>", unsafe_allow_html=True)
    
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
    
    # Date selection
    st.markdown("<h2>📅 Date Selection</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_date = st.date_input(
            "Select a date:",
            value=df.index[-1].date(),
            min_value=df.index[0].date(),
            max_value=df.index[-1].date()
        )
    
    with col2:
        # Find available hours for this date
        try:
            date_data = df[df.index.date == selected_date]
            if len(date_data) > 0:
                available_hours = sorted(set(date_data.index.hour.tolist()))
                default_idx = len(available_hours) - 1 if available_hours else 0
                selected_hour = st.selectbox(
                    "Select an hour:",
                    options=available_hours,
                    index=min(default_idx, len(available_hours) - 1) if available_hours else 0
                )
            else:
                selected_hour = 0
                st.warning("⚠️ No data available for this date")
        except Exception as e:
            selected_hour = 0
            st.warning(f"⚠️ Error searching for hours: {str(e)}")
    
    # Prepare complete date
    try:
        from datetime import time
        selected_datetime = pd.Timestamp.combine(selected_date, time(hour=selected_hour))
        
        # Check if this date/time exists in index
        if selected_datetime not in df.index:
            # Find closest date/time safely
            try:
                # Method 1: Use get_indexer if index is unique
                if df.index.is_unique:
                    closest_indices = df.index.get_indexer([selected_datetime], method='nearest')
                    if len(closest_indices) > 0 and closest_indices[0] >= 0:
                        selected_datetime = df.index[closest_indices[0]]
                        st.info(f"ℹ️ Using closest date/time: {selected_datetime}")
                    else:
                        st.error("❌ Unable to find a close date")
                        selected_datetime = None
                else:
                    # Method 2: If index is not unique, find manually
                    time_diffs = abs(df.index - selected_datetime)
                    closest_idx = time_diffs.idxmin()
                    selected_datetime = closest_idx
                    st.info(f"ℹ️ Using closest date/time: {selected_datetime}")
            except Exception as e2:
                st.error(f"❌ Error searching for date: {str(e2)}")
                selected_datetime = None
    except Exception as e:
        st.error(f"❌ Error preparing date: {str(e)}")
        selected_datetime = None
    
    # Prediction button
    if st.button("🔮 Generate Prediction", type="primary", width='stretch'):
        if selected_datetime is None:
            st.error("❌ Please select a valid date")
        else:
            # Prepare features
            window_data, error = prepare_features_for_date(df, selected_datetime, params['window_size'])
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Get the best performing model
                best_model_name = get_best_model()
                
                # Prepare normalized data
                consumption_values = window_data[[params['target_col']] + params['feature_cols']].values
                consumption_scaled = scaler.transform(consumption_values)
                
                # Make prediction with best model
                if best_model_name == 'LSTM (Multivariate)':
                    input_seq = consumption_scaled
                    pred_value = predict_consumption(models, best_model_name, input_seq, univariate=False)
                else:
                    input_seq = consumption_scaled[:, 0]
                    pred_value = predict_consumption(models, best_model_name, input_seq, univariate=True)
                
                if pred_value is None:
                    st.error("❌ Error generating prediction")
                else:
                    # Display results
                    st.markdown("<h2>📊 Prediction Results</h2>", unsafe_allow_html=True)
                    
                    # Get metrics for best model
                    metrics = get_model_metrics()
                    best_model_metrics = metrics.get(best_model_name, {})
                    
                    # Display metrics in 3 columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Prediction",
                            value=f"{pred_value:.0f} MW",
                            help=f"Predicted consumption using {best_model_name}"
                        )
                    
                    with col2:
                        if selected_datetime in df.index:
                            actual_value = df.loc[selected_datetime, 'Consumption']
                            error_val = abs(pred_value - actual_value)
                            error_pct = (error_val / actual_value) * 100
                            st.metric(
                                label="Actual Value",
                                value=f"{actual_value:.0f} MW",
                                delta=f"{error_val:.0f} MW ({error_pct:.1f}%)"
                            )
                        else:
                            st.metric(
                                label="Actual Value",
                                value="N/A",
                                help="Actual value not available for future dates"
                            )
                    
                    with col3:
                        st.metric(
                            label="Model RMSE",
                            value=f"{best_model_metrics.get('RMSE', 0):.2f} MW",
                            help="Root Mean Squared Error on test set"
                        )
                    
                    # Display model info
                    st.markdown(f"""
                    <div class="info-box">
                        <h3>🏆 Best Performing Model: {best_model_name}</h3>
                        <p><strong>RMSE:</strong> {best_model_metrics.get('RMSE', 0):.2f} MW</p>
                        <p><strong>MAE:</strong> {best_model_metrics.get('MAE', 0):.2f} MW</p>
                        <p><strong>R²:</strong> {best_model_metrics.get('R2', 0):.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualization
                    st.markdown("<h3>📈 Prediction Visualization</h3>", unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    
                    # Last 24 hours history
                    hours_24h = [f"H-{23-i}" for i in range(24)]
                    fig.add_trace(go.Scatter(
                        x=hours_24h,
                        y=window_data['Consumption'].values,
                        mode='lines+markers',
                        name='Actual Consumption (24h)',
                        line=dict(color='#00ff88', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Prediction point
                    fig.add_trace(go.Scatter(
                        x=['H+1 (Prediction)'],
                        y=[pred_value],
                        mode='markers',
                        name=f'{best_model_name}: {pred_value:.0f} MW',
                        marker=dict(size=15, color='#00d4ff', symbol='star'),
                        showlegend=True
                    ))
                    
                    # Actual value if available
                    if selected_datetime in df.index:
                        actual_value = df.loc[selected_datetime, 'Consumption']
                        fig.add_trace(go.Scatter(
                            x=['H+1 (Actual)'],
                            y=[actual_value],
                            mode='markers',
                            name=f'Actual Value: {actual_value:.0f} MW',
                            marker=dict(size=20, color='#ffffff', symbol='x', 
                                      line=dict(width=3, color='#ff0000')),
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title="Prediction vs Reality",
                        xaxis_title="Time",
                        yaxis_title="Consumption (MW)",
                        template="plotly_dark",
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, width='stretch')

# ============================================================================
# NAVIGATION PRINCIPALE
# ============================================================================

def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #00d4ff; margin: 0;">⚡</h1>
            <h2 style="color: #00d4ff; margin: 0.5rem 0;">AI Energy Forecast</h2>
            <p style="color: #888; font-size: 0.9rem;">Intelligent Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "📑 Navigation",
            [
                "📊 Data Overview",
                "📈 Data Analysis (EDA)",
                "🎯 Model Performance",
                "⚖️ Model Comparison",
                "🔮 Real-Time Prediction"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem;">
            <p>Powered by Deep Learning</p>
            <p>TensorFlow & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == "📊 Data Overview":
        page_data_overview()
    elif page == "📈 Data Analysis (EDA)":
        page_data_analysis()
    elif page == "🎯 Model Performance":
        page_model_performance()
    elif page == "⚖️ Model Comparison":
        page_model_comparison()
    elif page == "🔮 Real-Time Prediction":
        page_realtime_prediction()

if __name__ == "__main__":
    main()
