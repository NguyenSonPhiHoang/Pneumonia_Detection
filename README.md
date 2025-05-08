# Pneumonia Detection Project 

## Hướng dẫn cài đặt và chạy dự án

Bước 1: Tải dữ liệu (ChestXRay) 

Truy cập: 

    https://data.mendeley.com/datasets/rscbjbr9sj/2
Tải file: ChestXRay2017.zip

Tạo thư mục data/ trong dự án với cấu trúc sau:
data/

├── test/

│   ├── NORMAL/

│   └── PNEUMONIA/

└── train/

│   ├── NORMAL/

│   └── PNEUMONIA/

Giải nén ChestXRay2017.zip và đặt các hình ảnh vào các thư mục tương ứng.

Bước 2: Cài đặt Python 3.11 

Tải và cài đặt Python 3.11 để tránh xung đột thư viện: 

    https://www.python.org/downloads/release/python-3110/

Bước 3: Tạo môi trường ảo (venv)

Mở terminal/command prompt và di chuyển đến thư mục dự án.

Tạo môi trường ảo:

    python3.11 -m venv venv

Kích hoạt môi trường ảo:

    venv\Scripts\activate


Bước 4: Cài đặt các thư viện yêu cầu 
Chạy lệnh sau để cài đặt các thư viện cần thiết:

    pip install -r requirements.txt

Cách chạy dự án

Huấn luyện mô hình:
    
    python train.py

Đánh giá mô hình:

    python evaluate.py

Chạy ứng dụng web:

    python app.py

Sau đó, mở trình duyệt và truy cập http://localhost:5000.
