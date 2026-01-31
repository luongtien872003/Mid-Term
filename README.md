# 📊 Superstore Sales Dashboard - Big Data Midterm Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![Vaex](https://img.shields.io/badge/Vaex-Big_Data-purple)
![ML](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 📋 Giới thiệu

Bài tập giữa kỳ môn **Big Data** - Dashboard phân tích dữ liệu Superstore Sales với:
- **Vaex** cho xử lý Big Data
- **Machine Learning** cho dự đoán doanh thu
- **MongoDB Atlas** cho lưu trữ cloud

### 👥 Thành viên nhóm

| MSSV | Họ và Tên |
|------|-----------|
| K214162157 | Lương Minh Tiến |
| K214161343 | Lê Thành Tuân |

---

## 🎯 Tính năng chính

### 🚀 Big Data với Vaex
- Memory-mapped data processing
- Lazy evaluation cho hiệu suất cao
- Xử lý dataset lớn với RAM thấp

### 🤖 Machine Learning
- **Linear Regression** - Baseline model
- **Random Forest** - Ensemble learning
- **Gradient Boosting** - Advanced predictions
- Real-time Sales Prediction

### 📊 Interactive Dashboard
- Filter theo Category, Region, Segment
- KPI Metrics (Sales, Profit, Orders)
- Interactive Charts (Plotly)
- Data Table với search

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **MongoDB Atlas** | Cloud NoSQL Database |
| **Vaex** | Big Data Processing (thay Pandas) |
| **Scikit-learn** | Machine Learning |
| **Streamlit** | Web Dashboard |
| **Plotly** | Interactive Visualization |
| **Docker** | Containerization |
| **HuggingFace Spaces** | Cloud Deployment |

---

## 📁 Cấu trúc Project

```
Mid-Term/
├── 📓 notebook.ipynb      # Notebook phân tích (nộp LMS)
│                          # - Kết nối MongoDB
│                          # - Phân tích Vaex
│                          # - Visualization
│
├── 🐍 app.py              # Streamlit Dashboard
│                          # - Vaex Big Data
│                          # - ML Prediction
│                          # - Interactive UI
│
├── 📦 import_data.py      # Script import data
│
├── 🐳 Dockerfile          # Docker (Python 3.10 + Vaex)
│
├── 📋 requirements.txt    # Dependencies
│
└── 📖 README.md           # Documentation
```

---

## 🚀 Hướng dẫn chạy

### Yêu cầu
- **Python 3.10** (bắt buộc cho Vaex)
- Kết nối Internet

### 1. Clone & Setup

```bash
git clone https://github.com/luongtien872003/Mid-Term.git
cd Mid-Term

# Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy Streamlit

```bash
streamlit run app.py
```

Truy cập: http://localhost:8501

### 3. Chạy với Docker

```bash
docker build -t superstore-dashboard .
docker run -p 7860:7860 superstore-dashboard
```

Truy cập: http://localhost:7860

---

## 📊 Dataset

**Superstore Sales** - 10,000 bản ghi:

| Column | Mô tả |
|--------|-------|
| Order Date | Ngày đặt hàng |
| Category | Technology / Furniture / Office Supplies |
| Sub-Category | Phones, Chairs, Paper, ... |
| Region | East / West / Central / South |
| Segment | Consumer / Corporate / Home Office |
| Sales | Doanh thu ($) |
| Profit | Lợi nhuận ($) |
| Quantity | Số lượng |
| Discount | Chiết khấu |

---

## 🤖 Machine Learning Models

### Features sử dụng
- Category (encoded)
- Region (encoded)
- Segment (encoded)
- Sub-Category (encoded)
- Quantity
- Discount

### Target
- **Sales** (Doanh thu)

### Models

| Model | Mô tả |
|-------|-------|
| Linear Regression | Baseline, đơn giản |
| Random Forest | Ensemble 100 trees |
| Gradient Boosting | Sequential boosting |

### Metrics
- **R² Score** - Tỷ lệ variance explained
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error

---

## 📈 Vaex vs Pandas

| Tiêu chí | Vaex | Pandas |
|----------|------|--------|
| Memory Mapping | ✅ | ❌ |
| Lazy Evaluation | ✅ | ❌ |
| Out-of-core | ✅ | ❌ |
| 1 tỷ dòng | ✅ 8GB RAM | ❌ >100GB RAM |
| Parallel | ✅ Multi-thread | ❌ Single-thread |

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📁 GitHub | [github.com/luongtien872003/Mid-Term](https://github.com/luongtien872003/Mid-Term) |
| 🚀 Demo | [HuggingFace Spaces](https://huggingface.co/spaces/lmt872003/Mid-Term-Bigdata) |

---

## 📄 License

MIT License

---

**Thực hiện bởi:**
- 👨‍💻 **Lương Minh Tiến** – K214162157
- 👨‍💻 **Lê Thành Tuân** – K214161343

📅 **2024**
