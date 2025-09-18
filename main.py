
import cv2
import mediapipe as mp
import numpy as np
import pygame as pg
import random
from ultralytics import YOLO


# Khởi tạo nhận diện bàn tay
mp_ban_tay = mp.solutions.hands
nhan_dien_ban_tay = mp_ban_tay.Hands()
mp_ve = mp.solutions.drawing_utils

# Khởi tạo nhận diện khuôn mặt
mp_nhan_dien_mat = mp.solutions.face_detection


camera = cv2.VideoCapture(0)

# Khởi tạo YOLO
mo_hinh_yolo = YOLO('yolov8n.pt')  # Tự động tải xuống nếu chưa có

# Khởi tạo Pygame
TOC_DO_MIN = 0.25
TOC_DO_MAX = 5
SO_LUONG_BONG_MAX = 800
KICH_THUOC_MAN_HINH = pg.Vector2(800, 600)
BAN_KINH_HINH_TRON = 5

pg.init()
man_hinh_pygame = pg.display.set_mode(KICH_THUOC_MAN_HINH)
dong_ho_pygame = pg.time.Clock()

vi_tri_muc_tieu = None
danh_sach_bong = []

class Bong:
    def __init__(self, vi_tri, toc_do):
        self.vi_tri = vi_tri
        self.toc_do = toc_do

def dat_lai_bong():
    global danh_sach_bong, vi_tri_muc_tieu
    vi_tri_muc_tieu = None
    danh_sach_bong = []
    for x in range(SO_LUONG_BONG_MAX):
        vi_tri = pg.Vector2(
            random.randint(0, int(KICH_THUOC_MAN_HINH.x)), 
            random.randint(0, int(KICH_THUOC_MAN_HINH.y))
        )
        toc_do = random.uniform(TOC_DO_MIN, TOC_DO_MAX)
        danh_sach_bong.append(Bong(vi_tri, toc_do))

def tinh_trang_thai_ngon_tay(landmarks_ban_tay):
    trang_thai_ngon = []
    
    # Ngón cái (Thumb)
    dau_ngon_cai = landmarks_ban_tay.landmark[4].x
    goc_ngon_cai = landmarks_ban_tay.landmark[3].x
    trang_thai_ngon.append(1 if dau_ngon_cai < goc_ngon_cai else 0)
    
    # 4 ngón còn lại
    cac_dau = [8, 12, 16, 20]  # Đầu ngón
    cac_khop = [6, 10, 14, 18]  # Khớp giữa
    
    for dau, khop in zip(cac_dau, cac_khop):
        y_dau = landmarks_ban_tay.landmark[dau].y
        y_khop = landmarks_ban_tay.landmark[khop].y
        trang_thai_ngon.append(1 if y_dau < y_khop else 0)
    
    return trang_thai_ngon

def tao_mat_na_mo(frame, ket_qua_mat, ket_qua_tay):
    mat_na = np.zeros(frame.shape[:2], dtype=np.uint8)
    h, w = frame.shape[:2]
    
    # Tạo mask cho khuôn mặt
    if ket_qua_mat.detections:
        for phat_hien in ket_qua_mat.detections:
            hop = phat_hien.location_data.relative_bounding_box
            x = int(hop.xmin * w)
            y = int(hop.ymin * h)
            rong = int(hop.width * w)
            cao = int(hop.height * h)
            
            # Mở rộng vùng mask một chút
            le = 50
            x = max(0, x - le)
            y = max(0, y - le)
            rong = min(w - x, rong + 2 * le)
            cao = min(h - y, cao + 2 * le)
            
            cv2.rectangle(mat_na, (x, y), (x + rong, y + cao), 255, -1)
    
    # Tạo mask cho bàn tay
    if ket_qua_tay.multi_hand_landmarks:
        for landmarks_ban_tay in ket_qua_tay.multi_hand_landmarks:
            # Tìm bounding box của bàn tay
            x_coords = [landmark.x * w for landmark in landmarks_ban_tay.landmark]
            y_coords = [landmark.y * h for landmark in landmarks_ban_tay.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Mở rộng vùng mask
            le = 30
            x_min = max(0, x_min - le)
            y_min = max(0, y_min - le)
            x_max = min(w, x_max + le)
            y_max = min(h, y_max + le)
            
            cv2.rectangle(mat_na, (x_min, y_min), (x_max, y_max), 255, -1)
    
    return mat_na


def cap_nhat_pygame(vi_tri_ngon_tro):
    global vi_tri_muc_tieu
    for su_kien in pg.event.get():
        if su_kien.type == pg.QUIT:
            return False
        if su_kien.type == pg.KEYUP and su_kien.key == pg.K_r:
            dat_lai_bong()
    # Dùng vị trí ngón trỏ làm mục tiêu
    if vi_tri_ngon_tro:
        vi_tri_muc_tieu = vi_tri_ngon_tro
    man_hinh_pygame.fill((31, 143, 65))
    thoi_gian = dong_ho_pygame.tick(60)
    for bong in danh_sach_bong:
        if vi_tri_muc_tieu:
            bong.vi_tri.move_towards_ip(vi_tri_muc_tieu, bong.toc_do * thoi_gian)
        pg.draw.circle(man_hinh_pygame, (118, 207, 145), bong.vi_tri, BAN_KINH_HINH_TRON)
    # Vẽ mục tiêu
    if vi_tri_muc_tieu:
        pg.draw.circle(man_hinh_pygame, (255, 0, 0), vi_tri_muc_tieu, 10)
    pg.display.flip()
    pg.display.set_caption(f"fps: {round(dong_ho_pygame.get_fps(), 2)}, bong: {len(danh_sach_bong)}")
    return True


dat_lai_bong()

with mp_nhan_dien_mat.FaceDetection(min_detection_confidence=0.5) as nhan_khuonmat:
    while camera.isOpened():
        thanh_cong, khung_hinh = camera.read()
        if not thanh_cong:
            break

        # Lật ngược khung hình (flip theo trục dọc)
        khung_hinh = cv2.flip(khung_hinh, 1)

        # Chuyển ảnh sang RGB
        khung_hinh_rgb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)

        # Xử lý nhận diện khuôn mặt
        ket_qua_mat = nhan_khuonmat.process(khung_hinh_rgb)

        # Xử lý nhận diện bàn tay
        ket_qua_tay = nhan_dien_ban_tay.process(khung_hinh_rgb)

        # Chuyển ảnh lại về BGR để hiển thị
        khung_hinh = cv2.cvtColor(khung_hinh_rgb, cv2.COLOR_RGB2BGR)
        
        # Tạo hiệu ứng blur cho background
        khung_hinh_mo = cv2.GaussianBlur(khung_hinh, (21, 21), 0)
        
        # Tạo mask cho các vùng cần giữ nguyên (face + hands)
        mat_na = tao_mat_na_mo(khung_hinh, ket_qua_mat, ket_qua_tay)
        
        # Làm mềm edges của mask
        mat_na = cv2.GaussianBlur(mat_na, (5, 5), 0)
        
        # Chuyển mask thành 3 channels
        mat_na_3_kenh = cv2.cvtColor(mat_na, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Kết hợp frame gốc và frame blur
        khung_hinh_ket_qua = khung_hinh * mat_na_3_kenh + khung_hinh_mo * (1 - mat_na_3_kenh)
        khung_hinh_ket_qua = khung_hinh_ket_qua.astype(np.uint8)

        # YOLO object detection
        ket_qua_yolo = mo_hinh_yolo(khung_hinh, verbose=False)
        
        # Draw YOLO detections
        for ket_qua in ket_qua_yolo:
            hop = ket_qua.boxes
            if hop is not None:
                for hop_don in hop:
                    x1, y1, x2, y2 = map(int, hop_don.xyxy[0])
                    do_tin_cay = hop_don.conf[0]
                    lop = int(hop_don.cls[0])
                    
                    if do_tin_cay > 0.5:  # Confidence threshold
                        nhan = f'{mo_hinh_yolo.names[lop]}: {do_tin_cay:.2f}'
                        cv2.rectangle(khung_hinh_ket_qua, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(khung_hinh_ket_qua, nhan, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Khởi tạo trạng thái tay
        trang_thai_tay_trai = [0, 0, 0, 0, 0]
        trang_thai_tay_phai = [0, 0, 0, 0, 0]
        vi_tri_dau_ngon_tro = None

        # Vẽ khung nhận diện khuôn mặt
        if ket_qua_mat.detections:
            for phat_hien in ket_qua_mat.detections:
                mp_ve.draw_detection(khung_hinh_ket_qua, phat_hien)

        # Vẽ landmarks của bàn tay
        tim_thay_ngon_tro = False
        if ket_qua_tay.multi_hand_landmarks:
            for idx, landmarks_ban_tay in enumerate(ket_qua_tay.multi_hand_landmarks):
                mp_ve.draw_landmarks(khung_hinh_ket_qua, landmarks_ban_tay, mp_ban_tay.HAND_CONNECTIONS)
                # Tính trạng thái ngón tay
                trang_thai_ngon = tinh_trang_thai_ngon_tay(landmarks_ban_tay)
                # Get index finger tip position for pygame control
                dau_ngon_tro = landmarks_ban_tay.landmark[8]
                vi_tri_x = int(dau_ngon_tro.x * KICH_THUOC_MAN_HINH.x)
                vi_tri_y = int(dau_ngon_tro.y * KICH_THUOC_MAN_HINH.y)
                # Only use finger position if index finger is extended
                if trang_thai_ngon[1] == 1:  # Index finger extended
                    vi_tri_dau_ngon_tro = (vi_tri_x, vi_tri_y)
                    tim_thay_ngon_tro = True
                # Xác định tay trái hay phải
                nhan_tay = ket_qua_tay.multi_handedness[idx].classification[0].label
                if nhan_tay == "Left":
                    trang_thai_tay_phai = trang_thai_ngon
                else:
                    trang_thai_tay_trai = trang_thai_ngon
        # Nếu không phát hiện ngón trỏ, không có gì cần thiết lập
        if not tim_thay_ngon_tro:
            pass

        # Update pygame
        if not cap_nhat_pygame(vi_tri_dau_ngon_tro):
            break

        # Hiển thị trạng thái tay trên frame
        cv2.putText(khung_hinh_ket_qua, f"Tay trai: {trang_thai_tay_trai}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(khung_hinh_ket_qua, f"Tay phai: {trang_thai_tay_phai}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show finger control status
        if vi_tri_dau_ngon_tro:
            cv2.putText(khung_hinh_ket_qua, "Dang dieu khien bong bang ngon tay!", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Hiển thị hình ảnh
        cv2.imshow("Nhan dien Ban tay & Khuon mat + YOLO", khung_hinh_ket_qua)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
pg.quit()
