"""
📦 Import Data to MongoDB Atlas
Script này import dataset Superstore Sales vào MongoDB Atlas
Chạy script này trước khi chạy Streamlit app

Usage: python import_data.py
"""

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import certifi
import ssl
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# MongoDB Atlas URI với SSL
MONGODB_URI = "mongodb+srv://tienlm21416c:Tien872003@midterm.47arsdg.mongodb.net/?retryWrites=true&w=majority&appName=MidTerm"
DATABASE_NAME = "superstore_db"
COLLECTION_NAME = "sales"

def generate_sample_data(n_records=10000):
    """Generate sample Superstore Sales data"""
    print(f"📊 Đang tạo {n_records:,} bản ghi mẫu...")
    
    np.random.seed(42)
    
    categories = ['Technology', 'Furniture', 'Office Supplies']
    sub_categories = {
        'Technology': ['Phones', 'Computers', 'Accessories', 'Copiers'],
        'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
        'Office Supplies': ['Storage', 'Labels', 'Fasteners', 'Paper', 'Binders', 'Art', 'Envelopes', 'Appliances', 'Supplies']
    }
    regions = ['East', 'West', 'Central', 'South']
    states = ['California', 'New York', 'Texas', 'Pennsylvania', 'Illinois', 'Ohio', 'Florida', 'Washington', 'Georgia', 'North Carolina']
    cities = {
        'California': ['Los Angeles', 'San Francisco', 'San Diego', 'San Jose'],
        'New York': ['New York City', 'Buffalo', 'Rochester', 'Albany'],
        'Texas': ['Houston', 'Dallas', 'Austin', 'San Antonio'],
        'Pennsylvania': ['Philadelphia', 'Pittsburgh', 'Allentown'],
        'Illinois': ['Chicago', 'Aurora', 'Naperville'],
        'Ohio': ['Columbus', 'Cleveland', 'Cincinnati'],
        'Florida': ['Miami', 'Orlando', 'Tampa', 'Jacksonville'],
        'Washington': ['Seattle', 'Spokane', 'Tacoma'],
        'Georgia': ['Atlanta', 'Augusta', 'Savannah'],
        'North Carolina': ['Charlotte', 'Raleigh', 'Durham']
    }
    segments = ['Consumer', 'Corporate', 'Home Office']
    ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
    
    data = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(n_records):
        category = np.random.choice(categories)
        sub_category = np.random.choice(sub_categories[category])
        state = np.random.choice(states)
        city = np.random.choice(cities[state])
        
        order_date = base_date + timedelta(days=np.random.randint(0, 1460))  # 4 years
        ship_date = order_date + timedelta(days=np.random.randint(1, 7))
        
        # Realistic sales based on category
        if category == 'Technology':
            sales = round(np.random.exponential(400) + 50, 2)
        elif category == 'Furniture':
            sales = round(np.random.exponential(300) + 30, 2)
        else:
            sales = round(np.random.exponential(100) + 5, 2)
        
        quantity = np.random.randint(1, 15)
        discount = round(np.random.choice([0, 0.1, 0.15, 0.2, 0.3, 0.4]), 2)
        
        # Profit depends on discount
        profit_margin = np.random.uniform(0.1, 0.4) - discount
        profit = round(sales * profit_margin, 2)
        
        data.append({
            'Row ID': i + 1,
            'Order ID': f'US-{order_date.year}-{np.random.randint(100000, 999999)}',
            'Order Date': order_date.strftime('%Y-%m-%d'),
            'Ship Date': ship_date.strftime('%Y-%m-%d'),
            'Ship Mode': np.random.choice(ship_modes),
            'Customer ID': f'CG-{np.random.randint(10000, 99999)}',
            'Customer Name': f'Customer {np.random.randint(1, 1000)}',
            'Segment': np.random.choice(segments),
            'Country': 'United States',
            'City': city,
            'State': state,
            'Postal Code': np.random.randint(10000, 99999),
            'Region': np.random.choice(regions),
            'Product ID': f'{category[:3].upper()}-{sub_category[:2].upper()}-{np.random.randint(1000, 9999)}',
            'Category': category,
            'Sub-Category': sub_category,
            'Product Name': f'{sub_category} - {np.random.choice(["Premium", "Standard", "Basic", "Pro", "Elite"])} Model {np.random.randint(100, 999)}',
            'Sales': sales,
            'Quantity': quantity,
            'Discount': discount,
            'Profit': profit
        })
    
    return pd.DataFrame(data)

def connect_mongodb():
    """Connect to MongoDB Atlas"""
    print("🔗 Đang kết nối MongoDB Atlas...")
    
    # Thử nhiều cách kết nối
    connection_options = [
        # Option 1: Với certifi
        {
            "server_api": ServerApi('1'),
            "tlsCAFile": certifi.where(),
            "serverSelectionTimeoutMS": 30000,
            "connectTimeoutMS": 30000,
        },
        # Option 2: Allow invalid certificates (development only)
        {
            "server_api": ServerApi('1'),
            "tlsAllowInvalidCertificates": True,
            "serverSelectionTimeoutMS": 30000,
        },
        # Option 3: Simple connection
        {
            "serverSelectionTimeoutMS": 30000,
        }
    ]
    
    for i, options in enumerate(connection_options, 1):
        try:
            print(f"   Thử phương thức {i}...")
            client = MongoClient(MONGODB_URI, **options)
            client.admin.command('ping')
            print("✅ Kết nối MongoDB Atlas thành công!")
            return client
        except Exception as e:
            print(f"   ⚠️ Phương thức {i} thất bại: {str(e)[:50]}...")
            continue
    
    print("❌ Không thể kết nối MongoDB Atlas!")
    print("💡 Gợi ý: Kiểm tra firewall hoặc thử trên Google Colab")
    return None

def import_data(client, df):
    """Import data to MongoDB"""
    print(f"\n📥 Đang import {len(df):,} bản ghi vào MongoDB...")
    
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Drop existing collection
    collection.drop()
    print("🗑️ Đã xóa collection cũ (nếu có)")
    
    # Convert to list of dicts
    records = df.to_dict('records')
    
    # Insert data
    result = collection.insert_many(records)
    
    print(f"\n✅ Import thành công!")
    print(f"   📊 Số bản ghi: {len(result.inserted_ids):,}")
    print(f"   🗄️ Database: {DATABASE_NAME}")
    print(f"   📁 Collection: {COLLECTION_NAME}")
    
    return collection

def verify_data(collection):
    """Verify imported data"""
    print("\n" + "="*50)
    print("📋 KIỂM TRA DỮ LIỆU")
    print("="*50)
    
    # Count
    count = collection.count_documents({})
    print(f"\n📊 Tổng số bản ghi: {count:,}")
    
    # Sample
    print("\n📌 Mẫu 3 bản ghi:")
    for doc in collection.find().limit(3):
        print(f"   - Order: {doc['Order ID']} | {doc['Category']} | ${doc['Sales']:,.2f}")
    
    # Aggregation by Category
    print("\n📈 Thống kê theo Category:")
    pipeline = [
        {"$group": {
            "_id": "$Category",
            "total_sales": {"$sum": "$Sales"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"total_sales": -1}}
    ]
    
    for result in collection.aggregate(pipeline):
        print(f"   - {result['_id']}: ${result['total_sales']:,.2f} ({result['count']:,} đơn)")
    
    # Aggregation by Region
    print("\n🌍 Thống kê theo Region:")
    pipeline = [
        {"$group": {
            "_id": "$Region",
            "total_sales": {"$sum": "$Sales"},
            "total_profit": {"$sum": "$Profit"}
        }},
        {"$sort": {"total_sales": -1}}
    ]
    
    for result in collection.aggregate(pipeline):
        print(f"   - {result['_id']}: Sales ${result['total_sales']:,.2f} | Profit ${result['total_profit']:,.2f}")

def main():
    print("="*60)
    print("🚀 SUPERSTORE SALES - IMPORT DATA TO MONGODB")
    print("="*60)
    print(f"\n👥 Thực hiện: Lương Minh Tiến & Lê Thành Tuân")
    print(f"📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Generate sample data
    df = generate_sample_data(10000)
    print(f"✅ Đã tạo DataFrame với {len(df):,} dòng, {len(df.columns)} cột")
    
    # Connect to MongoDB
    client = connect_mongodb()
    if client is None:
        return
    
    # Import data
    collection = import_data(client, df)
    
    # Verify
    verify_data(collection)
    
    # Close connection
    client.close()
    print("\n✅ Đã đóng kết nối MongoDB")
    print("\n" + "="*60)
    print("🎉 HOÀN TẤT! Bây giờ có thể chạy Streamlit app")
    print("   streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    main()
