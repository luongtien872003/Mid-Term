# 📊 Superstore Sales Dashboard - Big Data Midterm Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

## 📋 Giới thiệu

Bài tập giữa kỳ môn **Big Data** - Phân tích dữ liệu Superstore Sales sử dụng các công nghệ Big Data hiện đại.

### 👥 Thành viên nhóm

| MSSV | Họ và Tên |
|------|-----------|
| K214162157 | Lương Minh Tiến |
| K214161343 | Lê Thành Tuân |

---

## 🎯 Mục tiêu dự án

1. **Kết nối và quản lý dữ liệu** với MongoDB Atlas (Cloud Database)
2. **Phân tích Big Data** sử dụng Vaex (thay thế Pandas)
3. **Xây dựng Dashboard tương tác** với Streamlit
4. **Triển khai ứng dụng** với Docker trên HuggingFace Spaces

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **MongoDB Atlas** | Cloud NoSQL Database - lưu trữ 10,000 bản ghi |
| **PyMongo** | Python driver cho MongoDB |
| **Vaex** | Xử lý Big Data hiệu quả (thay thế Pandas) |
| **Streamlit** | Web framework cho Data Dashboard |
| **Plotly** | Interactive visualization |
| **Docker** | Containerization |
| **HuggingFace Spaces** | Cloud deployment platform |

---

## 📁 Cấu trúc Project

```
Mid-Term/
├── 📓 notebook.ipynb      # Notebook chính (nộp LMS)
│                          # - Kết nối MongoDB Atlas
│                          # - Truy vấn và phân tích dữ liệu
│                          # - Visualization với Plotly
│
├── 🐍 app.py              # Streamlit Dashboard
│                          # - UI/UX professional (dark theme)
│                          # - Interactive filters
│                          # - Real-time charts
│
├── 📦 import_data.py      # Script import data vào MongoDB
│
├── 🐳 Dockerfile          # Docker configuration cho HuggingFace
│
├── 📋 requirements.txt    # Python dependencies
│
└── 📖 README.md           # Documentation (file này)
```

---

## 🚀 Hướng dẫn chạy

### Yêu cầu hệ thống

- Python 3.10+
- Kết nối Internet (để kết nối MongoDB Atlas)

### 1. Clone repository

```bash
git clone https://github.com/luongtien872003/Mid-Term.git
cd Mid-Term
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chạy Streamlit Dashboard

```bash
streamlit run app.py
```

Truy cập: http://localhost:8501

### 5. Chạy với Docker

```bash
# Build image
docker build -t superstore-dashboard .

# Run container
docker run -p 7860:7860 superstore-dashboard
```

Truy cập: http://localhost:7860

---

## 📊 Dataset

**Superstore Sales Dataset** - 10,000 bản ghi dữ liệu bán lẻ:

| Column | Mô tả |
|--------|-------|
| Order ID | Mã đơn hàng |
| Order Date | Ngày đặt hàng (2020-2023) |
| Ship Date | Ngày giao hàng |
| Customer ID/Name | Thông tin khách hàng |
| Segment | Consumer / Corporate / Home Office |
| Region | East / West / Central / South |
| Category | Technology / Furniture / Office Supplies |
| Sub-Category | Phones, Chairs, Paper, ... |
| Sales | Doanh thu ($) |
| Profit | Lợi nhuận ($) |
| Quantity | Số lượng |
| Discount | Chiết khấu (%) |

---

## 📈 Tính năng Dashboard

### 🎛️ Bộ lọc (Filters)
- Danh mục sản phẩm (Category)
- Khu vực (Region)
- Phân khúc khách hàng (Segment)
- Khoảng thời gian (Date Range)

### 📊 Biểu đồ (Charts)
- **KPI Cards**: Tổng doanh thu, lợi nhuận, số đơn hàng
- **Bar Chart**: Doanh thu & Lợi nhuận theo Category
- **Pie Chart**: Phân bố theo Region
- **Line Chart**: Xu hướng theo thời gian
- **Horizontal Bar**: Top 10 sản phẩm bán chạy
- **Heatmap**: Sub-Category x Region

### 📋 Data Table
- Tìm kiếm sản phẩm
- Hiển thị 100 bản ghi
- Format columns

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📁 GitHub Repository | [github.com/luongtien872003/Mid-Term](https://github.com/luongtien872003/Mid-Term) |
| 🚀 Demo Online | [HuggingFace Spaces](https://huggingface.co/spaces/lmt872003/Mid-Term-Bigdata) |

---

## 📝 Tại sao Vaex phù hợp Big Data hơn Pandas?

| Tiêu chí | Vaex | Pandas |
|----------|------|--------|
| Memory mapping | ✅ Có | ❌ Không |
| Lazy evaluation | ✅ Có | ❌ Không |
| Out-of-core processing | ✅ Có | ❌ Không |
| Xử lý 1 tỷ dòng | ✅ Laptop 8GB RAM | ❌ Cần > 100GB RAM |
| Parallel processing | ✅ Multi-threaded | ❌ Single-threaded |

**Kết luận**: Vaex phù hợp hơn cho Big Data vì sử dụng memory-mapped files và lazy evaluation, cho phép xử lý datasets lớn gấp nhiều lần RAM khả dụng.

---

## 🐳 HuggingFace Deployment

Project được deploy trên HuggingFace Spaces với Docker SDK:

```yaml
title: Superstore Sales Dashboard
emoji: 📊
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
```

---

## 📄 License

MIT License - Free to use for educational purposes.

---

**Thực hiện bởi:**
- 👨‍💻 **Lương Minh Tiến** – K214162157
- 👨‍💻 **Lê Thành Tuân** – K214161343

📅 **2024**
