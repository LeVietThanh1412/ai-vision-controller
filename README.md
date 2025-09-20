# AI Vision Controller

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google)
![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple?logo=yolo)
![Pygame](https://img.shields.io/badge/Pygame-2.6-green?logo=pygame)
![Numpy](https://img.shields.io/badge/Numpy-1.26-blue?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Giới thiệu
AI Vision Controller là ứng dụng nhận diện bàn tay và khuôn mặt bằng AI, kết hợp với Pygame để điều khiển các đối tượng trên màn hình theo chuyển động thực tế.

## Công nghệ sử dụng
- **Python 3.12**
- **OpenCV**: Xử lý hình ảnh từ camera.
- **MediaPipe**: Nhận diện bàn tay và khuôn mặt.
- **Ultralytics YOLOv8**: Nhận diện đối tượng nâng cao.
- **Pygame**: Hiển thị và điều khiển giao diện đồ họa.

## Cài đặt
1. Clone repo về máy:
    ```bash
    git clone https://github.com/LeVietThanh1412/ai-vision-controller.git
    ```
2. Tạo môi trường ảo và cài đặt các thư viện:
    ```bash
    python -m venv venv
    venv\Scripts\activate   # Windows
    source venv/bin/activate # Ubuntu

    pip install -r requirements.txt
    ```

## Sử dụng
1. Kết nối webcam với máy tính.
2. Chạy ứng dụng:
    ```bash
    python main.py
    ```
3. Dùng bàn tay và khuôn mặt để điều khiển các đối tượng trên màn hình Pygame.

## Lưu ý
- Không upload thư mục môi trường ảo và file model lớn lên GitHub (đã cấu hình trong `.gitignore`).
- Nếu thiếu file model YOLO (`yolov8n.pt`), ứng dụng sẽ tự động tải về.

## Tác giả
LeVietThanh1412