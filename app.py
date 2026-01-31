"""
🎯 Superstore Sales Dashboard with Vaex & Machine Learning
Big Data Midterm Project - Streamlit Application

Authors:
- Lương Minh Tiến – K214162157
- Lê Thành Tuân – K214161343

Features:
- Vaex for Big Data processing
- Machine Learning for Sales Prediction
- Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import warnings
import pickle
import io

warnings.filterwarnings('ignore')

# Try to import Vaex
try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False

# Try to import sklearn for ML
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Superstore Sales Dashboard - Big Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e94560 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00d9ff !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #00ff88 !important;
    }
    
    /* Success box */
    .success-box {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* Info box */
    .info-box {
        background: rgba(0, 217, 255, 0.1);
        border-left: 4px solid #00d9ff;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* ML box */
    .ml-box {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Title animation */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-title {
        background: linear-gradient(90deg, #00d9ff, #e94560, #00d9ff);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Vaex badge */
    .vaex-badge {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 5px;
    }
    
    .ml-badge {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MONGODB CONNECTION
# =============================================================================
@st.cache_resource
def get_mongodb_connection():
    """Create MongoDB connection"""
    import certifi
    
    uri = "mongodb+srv://tienlm21416c:Tien872003@midterm.47arsdg.mongodb.net/?retryWrites=true&w=majority&appName=MidTerm"
    
    connection_options = [
        {"server_api": ServerApi('1'), "tlsCAFile": certifi.where()},
        {"server_api": ServerApi('1'), "tlsAllowInvalidCertificates": True},
        {},
    ]
    
    for options in connection_options:
        try:
            client = MongoClient(uri, **options)
            client.admin.command('ping')
            return client
        except:
            continue
    
    return None

@st.cache_data(ttl=300)
def load_data_from_mongodb():
    """Load data from MongoDB and convert to Vaex if available"""
    client = get_mongodb_connection()
    if client is None:
        return None, None
    
    try:
        db = client['superstore_db']
        collection = db['sales']
        
        cursor = collection.find({})
        data = list(cursor)
        
        if len(data) == 0:
            return None, None
        
        # Convert to Pandas DataFrame first
        df = pd.DataFrame(data)
        
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Convert date columns
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
        if 'Ship Date' in df.columns:
            df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        # Convert to Vaex DataFrame if available
        vdf = None
        if VAEX_AVAILABLE:
            try:
                vdf = vaex.from_pandas(df)
            except:
                vdf = None
        
        return df, vdf
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None

# =============================================================================
# VAEX ANALYSIS FUNCTIONS
# =============================================================================
def analyze_with_vaex(vdf, df, group_col, agg_col='Sales'):
    """Perform analysis using Vaex (or Pandas fallback)"""
    if VAEX_AVAILABLE and vdf is not None:
        # Use Vaex
        result = vdf.groupby(group_col, agg={
            'total': vaex.agg.sum(agg_col),
            'count': vaex.agg.count(agg_col),
            'mean': vaex.agg.mean(agg_col)
        })
        return result.to_pandas_df()
    else:
        # Pandas fallback
        result = df.groupby(group_col).agg(
            total=(agg_col, 'sum'),
            count=(agg_col, 'count'),
            mean=(agg_col, 'mean')
        ).reset_index()
        return result

def get_stats_with_vaex(vdf, df, col):
    """Get statistics using Vaex"""
    if VAEX_AVAILABLE and vdf is not None:
        return {
            'min': float(vdf[col].min()),
            'max': float(vdf[col].max()),
            'mean': float(vdf[col].mean()),
            'std': float(vdf[col].std()),
            'sum': float(vdf[col].sum())
        }
    else:
        return {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'sum': float(df[col].sum())
        }

# =============================================================================
# MACHINE LEARNING FUNCTIONS
# =============================================================================
@st.cache_resource
def train_ml_models(_df):
    """Train Machine Learning models for Sales prediction"""
    if not ML_AVAILABLE:
        return None, None, None
    
    df = _df.copy()
    
    # Prepare features
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    le_segment = LabelEncoder()
    le_subcategory = LabelEncoder()
    
    df['Category_encoded'] = le_category.fit_transform(df['Category'])
    df['Region_encoded'] = le_region.fit_transform(df['Region'])
    df['Segment_encoded'] = le_segment.fit_transform(df['Segment'])
    df['SubCategory_encoded'] = le_subcategory.fit_transform(df['Sub-Category'])
    
    # Features and target
    feature_cols = ['Category_encoded', 'Region_encoded', 'Segment_encoded', 
                    'SubCategory_encoded', 'Quantity', 'Discount']
    X = df[feature_cols]
    y = df['Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['Linear Regression'] = {
        'model': lr,
        'r2': r2_score(y_test, lr_pred),
        'mae': mean_absolute_error(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['Random Forest'] = {
        'model': rf,
        'r2': r2_score(y_test, rf_pred),
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
    }
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    models['Gradient Boosting'] = {
        'model': gb,
        'r2': r2_score(y_test, gb_pred),
        'mae': mean_absolute_error(y_test, gb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred))
    }
    
    # Encoders for prediction
    encoders = {
        'Category': le_category,
        'Region': le_region,
        'Segment': le_segment,
        'Sub-Category': le_subcategory
    }
    
    return models, encoders, (X_test, y_test)

def predict_sales(models, encoders, category, region, segment, subcategory, quantity, discount):
    """Predict sales using trained models"""
    if not models:
        return None
    
    # Encode inputs
    try:
        cat_enc = encoders['Category'].transform([category])[0]
        reg_enc = encoders['Region'].transform([region])[0]
        seg_enc = encoders['Segment'].transform([segment])[0]
        sub_enc = encoders['Sub-Category'].transform([subcategory])[0]
    except:
        return None
    
    X_new = np.array([[cat_enc, reg_enc, seg_enc, sub_enc, quantity, discount]])
    
    predictions = {}
    for name, model_info in models.items():
        pred = model_info['model'].predict(X_new)[0]
        predictions[name] = max(0, pred)  # Sales can't be negative
    
    return predictions

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    """Render sidebar with filters"""
    st.sidebar.markdown("## 🎛️ Bộ lọc dữ liệu")
    
    # Show tech badges
    if VAEX_AVAILABLE:
        st.sidebar.markdown('<span class="vaex-badge">✅ Vaex Active</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="vaex-badge">⚠️ Vaex Fallback</span>', unsafe_allow_html=True)
    
    if ML_AVAILABLE:
        st.sidebar.markdown('<span class="ml-badge">🤖 ML Active</span>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Category filter
    categories = ['Tất cả'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("📦 Danh mục", categories, index=0)
    
    # Region filter
    regions = ['Tất cả'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("🌍 Khu vực", regions, index=0)
    
    # Segment filter
    segments = ['Tất cả'] + sorted(df['Segment'].unique().tolist())
    selected_segment = st.sidebar.selectbox("👥 Phân khúc", segments, index=0)
    
    # Date range filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("📅 **Khoảng thời gian**")
    
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Chọn khoảng ngày",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ Thông tin")
    st.sidebar.info("""
    **Bài tập Giữa kỳ Big Data**
    
    👨‍💻 Thực hiện:
    - Lương Minh Tiến
    - Lê Thành Tuân
    
    📚 Công nghệ:
    - MongoDB Atlas
    - Vaex (Big Data)
    - Scikit-learn (ML)
    - Streamlit
    """)
    
    return selected_category, selected_region, selected_segment, date_range

def filter_data(df, vdf, category, region, segment, date_range):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    if category != 'Tất cả':
        filtered_df = filtered_df[filtered_df['Category'] == category]
    
    if region != 'Tất cả':
        filtered_df = filtered_df[filtered_df['Region'] == region]
    
    if segment != 'Tất cả':
        filtered_df = filtered_df[filtered_df['Segment'] == segment]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Order Date'].dt.date >= start_date) &
            (filtered_df['Order Date'].dt.date <= end_date)
        ]
    
    # Convert filtered to Vaex
    filtered_vdf = None
    if VAEX_AVAILABLE:
        try:
            filtered_vdf = vaex.from_pandas(filtered_df)
        except:
            pass
    
    return filtered_df, filtered_vdf

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def render_kpi_metrics(df, vdf):
    """Render KPI metrics using Vaex"""
    stats_sales = get_stats_with_vaex(vdf, df, 'Sales')
    stats_profit = get_stats_with_vaex(vdf, df, 'Profit')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Tổng Doanh Thu",
            value=f"${stats_sales['sum']:,.0f}",
            delta=f"{len(df):,} đơn hàng"
        )
    
    with col2:
        profit_margin = (stats_profit['sum'] / stats_sales['sum']) * 100 if stats_sales['sum'] > 0 else 0
        st.metric(
            label="📈 Tổng Lợi Nhuận",
            value=f"${stats_profit['sum']:,.0f}",
            delta=f"{profit_margin:.1f}% margin"
        )
    
    with col3:
        st.metric(
            label="🛒 Giá trị TB/Đơn",
            value=f"${stats_sales['mean']:,.0f}",
            delta=f"Std: ${stats_sales['std']:,.0f}"
        )
    
    with col4:
        avg_discount = df['Discount'].mean() * 100
        st.metric(
            label="🏷️ Chiết khấu TB",
            value=f"{avg_discount:.1f}%",
            delta="của giá gốc"
        )

def render_category_chart(df, vdf):
    """Sales by Category using Vaex"""
    result = analyze_with_vaex(vdf, df, 'Category', 'Sales')
    
    fig = px.bar(
        result,
        x='Category',
        y='total',
        color='Category',
        title='📊 Doanh Thu theo Category (Vaex Analysis)',
        labels={'total': 'Doanh thu ($)', 'Category': 'Danh mục'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    
    return fig

def render_region_chart(df, vdf):
    """Profit by Region using Vaex"""
    result = analyze_with_vaex(vdf, df, 'Region', 'Profit')
    
    fig = px.pie(
        result,
        values='total',
        names='Region',
        title='🗺️ Lợi Nhuận theo Region (Vaex Analysis)',
        color_discrete_sequence=px.colors.sequential.Plasma,
        hole=0.4
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def render_trend_chart(df):
    """Sales trend over time"""
    df_time = df.copy()
    df_time['YearMonth'] = df_time['Order Date'].dt.to_period('M').astype(str)
    
    sales_trend = df_time.groupby('YearMonth').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sales_trend['YearMonth'],
        y=sales_trend['Sales'],
        mode='lines+markers',
        name='Doanh thu',
        line=dict(color='#00d9ff', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=sales_trend['YearMonth'],
        y=sales_trend['Profit'],
        mode='lines+markers',
        name='Lợi nhuận',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='📈 Xu Hướng Doanh Thu & Lợi Nhuận',
        xaxis_title='Thời gian',
        yaxis_title='Số tiền (USD)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig

def render_top_products(df, vdf, n=10):
    """Top products using Vaex"""
    result = analyze_with_vaex(vdf, df, 'Product Name', 'Sales')
    top_n = result.nlargest(n, 'total')
    
    fig = px.bar(
        top_n,
        x='total',
        y='Product Name',
        orientation='h',
        title=f'🏆 Top {n} Sản Phẩm (Vaex Analysis)',
        color='total',
        color_continuous_scale='Blues',
        labels={'total': 'Doanh thu ($)', 'Product Name': 'Sản phẩm'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False,
        height=400
    )
    
    return fig

# =============================================================================
# ML PREDICTION SECTION
# =============================================================================
def render_ml_section(df, models, encoders):
    """Render Machine Learning prediction section"""
    st.markdown("---")
    st.markdown("## 🤖 Machine Learning - Dự đoán Doanh Thu")
    
    if not ML_AVAILABLE or not models:
        st.warning("⚠️ Machine Learning không khả dụng. Cài đặt scikit-learn để sử dụng tính năng này.")
        return
    
    # Model Performance
    st.markdown("### 📊 Hiệu suất các mô hình")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (name, info) in enumerate(models.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="ml-box">
                <h4>{name}</h4>
                <p>R² Score: <b>{info['r2']:.3f}</b></p>
                <p>MAE: <b>${info['mae']:,.2f}</b></p>
                <p>RMSE: <b>${info['rmse']:,.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction form
    st.markdown("### 🔮 Dự đoán Doanh Thu cho đơn hàng mới")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_category = st.selectbox("📦 Category", df['Category'].unique())
        pred_region = st.selectbox("🌍 Region", df['Region'].unique())
    
    with col2:
        pred_segment = st.selectbox("👥 Segment", df['Segment'].unique())
        pred_subcategory = st.selectbox("📂 Sub-Category", df['Sub-Category'].unique())
    
    with col3:
        pred_quantity = st.slider("📊 Quantity", 1, 20, 5)
        pred_discount = st.slider("🏷️ Discount", 0.0, 0.5, 0.1, 0.05)
    
    if st.button("🚀 Dự đoán Sales", type="primary"):
        predictions = predict_sales(
            models, encoders,
            pred_category, pred_region, pred_segment, pred_subcategory,
            pred_quantity, pred_discount
        )
        
        if predictions:
            st.markdown("### 📈 Kết quả dự đoán:")
            
            cols = st.columns(len(predictions))
            for i, (model_name, pred_value) in enumerate(predictions.items()):
                with cols[i]:
                    st.metric(
                        label=model_name,
                        value=f"${pred_value:,.2f}"
                    )
            
            # Best model recommendation
            best_model = max(models.items(), key=lambda x: x[1]['r2'])[0]
            st.success(f"💡 **Khuyến nghị**: Sử dụng {best_model} (R² cao nhất: {models[best_model]['r2']:.3f})")
        else:
            st.error("❌ Không thể dự đoán. Vui lòng kiểm tra dữ liệu đầu vào.")

# =============================================================================
# DATA TABLE
# =============================================================================
def render_data_table(df):
    """Render data table"""
    st.markdown("### 📋 Chi tiết đơn hàng")
    
    display_cols = ['Order Date', 'Customer Name', 'Category', 'Sub-Category', 
                    'Product Name', 'Region', 'Sales', 'Profit', 'Quantity', 'Discount']
    available_cols = [col for col in display_cols if col in df.columns]
    
    search = st.text_input("🔍 Tìm kiếm", "")
    
    display_df = df[available_cols].copy()
    
    if search:
        mask = display_df.apply(lambda x: x.astype(str).str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]
    
    if 'Order Date' in display_df.columns:
        display_df['Order Date'] = display_df['Order Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df.head(100), use_container_width=True, height=400)
    st.caption(f"Hiển thị {min(100, len(display_df))} / {len(display_df)} bản ghi")

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Title
    st.markdown('<h1 class="main-title">📊 Superstore Sales Dashboard</h1>', unsafe_allow_html=True)
    
    # Tech badges
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        badges = ""
        if VAEX_AVAILABLE:
            badges += '<span class="vaex-badge">🚀 Vaex Big Data</span>'
        if ML_AVAILABLE:
            badges += '<span class="ml-badge">🤖 Machine Learning</span>'
        st.markdown(f'<div style="text-align: center">{badges}</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🔄 Đang kết nối MongoDB và tải dữ liệu...'):
        df, vdf = load_data_from_mongodb()
    
    if df is None or len(df) == 0:
        st.error("""
        ❌ **Không thể tải dữ liệu từ MongoDB!**
        
        Vui lòng chạy script import_data.py để import dữ liệu vào MongoDB.
        """)
        st.stop()
    
    # Success message
    vaex_status = "Vaex" if VAEX_AVAILABLE else "Pandas (fallback)"
    st.markdown(f"""
    <div class="success-box">
        ✅ <strong>Kết nối thành công!</strong> Đã tải {len(df):,} bản ghi | Engine: {vaex_status}
    </div>
    """, unsafe_allow_html=True)
    
    # Train ML models
    if ML_AVAILABLE:
        with st.spinner('🤖 Đang huấn luyện mô hình ML...'):
            models, encoders, test_data = train_ml_models(df)
    else:
        models, encoders = None, None
    
    # Sidebar filters
    category, region, segment, date_range = render_sidebar(df)
    
    # Apply filters
    filtered_df, filtered_vdf = filter_data(df, vdf, category, region, segment, date_range)
    
    # Show filter status
    if len(filtered_df) < len(df):
        st.info(f"🔍 Đang hiển thị {len(filtered_df):,} / {len(df):,} bản ghi")
    
    # KPI Metrics
    st.markdown("---")
    render_kpi_metrics(filtered_df, filtered_vdf)
    
    # Charts - Row 1
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_category_chart(filtered_df, filtered_vdf), use_container_width=True)
    
    with col2:
        st.plotly_chart(render_region_chart(filtered_df, filtered_vdf), use_container_width=True)
    
    # Trend Chart
    st.plotly_chart(render_trend_chart(filtered_df), use_container_width=True)
    
    # Top Products
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(render_top_products(filtered_df, filtered_vdf), use_container_width=True)
    
    # ML Section
    if ML_AVAILABLE:
        render_ml_section(df, models, encoders)
    
    # Data Table
    st.markdown("---")
    with st.expander("📋 Xem Chi Tiết Dữ Liệu"):
        render_data_table(filtered_df)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎓 <strong>Bài tập Giữa kỳ Big Data</strong></p>
        <p>MongoDB Atlas • Vaex • Scikit-learn • Streamlit • Docker</p>
        <p>👨‍💻 Lương Minh Tiến (K214162157) • Lê Thành Tuân (K214161343)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
