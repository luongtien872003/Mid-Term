"""
🎯 Superstore Sales Dashboard
Big Data Midterm Project - Streamlit Application

Authors:
- Lương Minh Tiến – K214162157
- Lê Thành Tuân – K214161343
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Superstore Sales Dashboard",
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
    
    /* Cards styling */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
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
    
    /* Success box */
    .success-box {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* Stacked bar animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d9ff, #e94560);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MONGODB CONNECTION
# =============================================================================
@st.cache_resource
def get_mongodb_connection():
    """Create MongoDB connection"""
    import certifi
    from pymongo.server_api import ServerApi
    
    uri = "mongodb+srv://tienlm21416c:Tien872003@midterm.47arsdg.mongodb.net/?retryWrites=true&w=majority&appName=MidTerm"
    
    # Thử nhiều cách kết nối
    connection_options = [
        # Option 1: Với certifi
        {"server_api": ServerApi('1'), "tlsCAFile": certifi.where()},
        # Option 2: Allow invalid certificates
        {"server_api": ServerApi('1'), "tlsAllowInvalidCertificates": True},
        # Option 3: Simple
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_mongodb():
    """Load data from MongoDB"""
    client = get_mongodb_connection()
    if client is None:
        return None
    
    try:
        db = client['superstore_db']
        collection = db['sales']
        
        # Get all documents
        cursor = collection.find({})
        data = list(cursor)
        
        if len(data) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Remove MongoDB _id
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Convert date columns
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
        if 'Ship Date' in df.columns:
            df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    """Render sidebar with filters"""
    st.sidebar.markdown("## 🎛️ Bộ lọc dữ liệu")
    st.sidebar.markdown("---")
    
    # Category filter
    categories = ['Tất cả'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox(
        "📦 Danh mục sản phẩm",
        categories,
        index=0
    )
    
    # Region filter
    regions = ['Tất cả'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox(
        "🌍 Khu vực",
        regions,
        index=0
    )
    
    # Segment filter
    segments = ['Tất cả'] + sorted(df['Segment'].unique().tolist())
    selected_segment = st.sidebar.selectbox(
        "👥 Phân khúc khách hàng",
        segments,
        index=0
    )
    
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
    - Vaex / Pandas
    - Streamlit
    - Docker
    """)
    
    return selected_category, selected_region, selected_segment, date_range

def filter_data(df, category, region, segment, date_range):
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
    
    return filtered_df

# =============================================================================
# METRICS
# =============================================================================
def render_kpi_metrics(df):
    """Render KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = len(df)
    avg_discount = df['Discount'].mean() * 100
    
    with col1:
        st.metric(
            label="💰 Tổng Doanh Thu",
            value=f"${total_sales:,.0f}",
            delta=f"{total_orders:,} đơn hàng"
        )
    
    with col2:
        profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
        st.metric(
            label="📈 Tổng Lợi Nhuận",
            value=f"${total_profit:,.0f}",
            delta=f"{profit_margin:.1f}% margin"
        )
    
    with col3:
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        st.metric(
            label="🛒 Giá trị TB/Đơn",
            value=f"${avg_order_value:,.0f}",
            delta=f"{df['Quantity'].sum():,} sản phẩm"
        )
    
    with col4:
        st.metric(
            label="🏷️ Chiết khấu TB",
            value=f"{avg_discount:.1f}%",
            delta="của giá gốc"
        )

# =============================================================================
# CHARTS
# =============================================================================
def render_sales_by_category(df):
    """Sales by Category chart"""
    sales_by_cat = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = px.bar(
        sales_by_cat,
        x='Category',
        y=['Sales', 'Profit'],
        title='📊 Doanh Thu & Lợi Nhuận theo Danh Mục',
        barmode='group',
        color_discrete_sequence=['#00d9ff', '#e94560'],
        labels={'value': 'Số tiền (USD)', 'Category': 'Danh mục', 'variable': 'Loại'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def render_sales_by_region(df):
    """Sales by Region pie chart"""
    sales_by_region = df.groupby('Region')['Sales'].sum().reset_index()
    
    fig = px.pie(
        sales_by_region,
        values='Sales',
        names='Region',
        title='🗺️ Phân Bố Doanh Thu theo Khu Vực',
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

def render_sales_trend(df):
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
        title='📈 Xu Hướng Doanh Thu & Lợi Nhuận Theo Thời Gian',
        xaxis_title='Thời gian',
        yaxis_title='Số tiền (USD)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig

def render_top_products(df, n=10):
    """Top N products by sales"""
    top_products = df.groupby('Product Name')['Sales'].sum().nlargest(n).reset_index()
    
    fig = px.bar(
        top_products,
        x='Sales',
        y='Product Name',
        orientation='h',
        title=f'🏆 Top {n} Sản Phẩm Theo Doanh Thu',
        color='Sales',
        color_continuous_scale='Blues',
        labels={'Sales': 'Doanh thu (USD)', 'Product Name': 'Tên sản phẩm'}
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

def render_segment_analysis(df):
    """Customer segment analysis"""
    segment_data = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    segment_data.columns = ['Segment', 'Sales', 'Profit', 'Orders']
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=('Doanh thu theo Segment', 'Tỷ lệ đơn hàng')
    )
    
    fig.add_trace(
        go.Bar(
            x=segment_data['Segment'],
            y=segment_data['Sales'],
            marker_color=['#00d9ff', '#e94560', '#00ff88'],
            name='Doanh thu'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=segment_data['Segment'],
            values=segment_data['Orders'],
            marker_colors=['#00d9ff', '#e94560', '#00ff88'],
            name='Đơn hàng'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='👥 Phân Tích Segment Khách Hàng',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False,
        height=400
    )
    
    return fig

def render_subcategory_heatmap(df):
    """Subcategory performance heatmap"""
    pivot = df.pivot_table(
        values='Sales',
        index='Sub-Category',
        columns='Region',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot,
        title='🔥 Heatmap Doanh Thu: Sub-Category x Region',
        color_continuous_scale='RdYlBu_r',
        labels={'color': 'Doanh thu (USD)'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500
    )
    
    return fig

# =============================================================================
# DATA TABLE
# =============================================================================
def render_data_table(df):
    """Render interactive data table"""
    st.markdown("### 📋 Chi tiết đơn hàng")
    
    # Column selection
    display_cols = ['Order Date', 'Customer Name', 'Category', 'Sub-Category', 
                    'Product Name', 'Region', 'Sales', 'Profit', 'Quantity', 'Discount']
    available_cols = [col for col in display_cols if col in df.columns]
    
    # Search box
    search = st.text_input("🔍 Tìm kiếm sản phẩm", "")
    
    display_df = df[available_cols].copy()
    
    if search:
        mask = display_df.apply(lambda x: x.astype(str).str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]
    
    # Format columns
    if 'Order Date' in display_df.columns:
        display_df['Order Date'] = display_df['Order Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=400
    )
    
    st.caption(f"Hiển thị {min(100, len(display_df))} / {len(display_df)} bản ghi")

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Title with animation
    st.markdown('<h1 class="main-title">📊 Superstore Sales Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🔄 Đang kết nối MongoDB và tải dữ liệu...'):
        df = load_data_from_mongodb()
    
    if df is None or len(df) == 0:
        st.error("""
        ❌ **Không thể tải dữ liệu từ MongoDB!**
        
        Vui lòng chạy notebook.ipynb để import dữ liệu vào MongoDB trước.
        
        Các bước:
        1. Mở Google Colab hoặc Jupyter
        2. Chạy notebook.ipynb
        3. Quay lại dashboard này
        """)
        st.stop()
    
    # Success message
    st.markdown(f"""
    <div class="success-box">
        ✅ <strong>Kết nối thành công!</strong> Đã tải {len(df):,} bản ghi từ MongoDB Atlas
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    category, region, segment, date_range = render_sidebar(df)
    
    # Apply filters
    filtered_df = filter_data(df, category, region, segment, date_range)
    
    # Show filter status
    if len(filtered_df) < len(df):
        st.info(f"🔍 Đang hiển thị {len(filtered_df):,} / {len(df):,} bản ghi theo bộ lọc")
    
    # KPI Metrics
    st.markdown("---")
    render_kpi_metrics(filtered_df)
    
    # Charts - Row 1
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_sales_by_category(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(render_sales_by_region(filtered_df), use_container_width=True)
    
    # Charts - Row 2
    st.plotly_chart(render_sales_trend(filtered_df), use_container_width=True)
    
    # Charts - Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_top_products(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(render_segment_analysis(filtered_df), use_container_width=True)
    
    # Heatmap
    with st.expander("🔥 Xem Heatmap Chi Tiết"):
        st.plotly_chart(render_subcategory_heatmap(filtered_df), use_container_width=True)
    
    # Data Table
    st.markdown("---")
    with st.expander("📋 Xem Chi Tiết Dữ Liệu"):
        render_data_table(filtered_df)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎓 <strong>Bài tập Giữa kỳ Big Data</strong> | 
        MongoDB Atlas • Vaex • Streamlit • Docker</p>
        <p>👨‍💻 Lương Minh Tiến (K214162157) • Lê Thành Tuân (K214161343)</p>
        <p>📅 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
