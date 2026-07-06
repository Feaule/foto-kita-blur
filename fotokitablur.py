import cv2
import mediapipe as mp
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

# Mengatur resolusi kamera menjadi 16:9 (contoh: 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

def is_peace_sign(hand_landmarks):
    """
    Mendeteksi apakah pose tangan adalah pose 'peace' (jari telunjuk dan tengah berdiri, 
    jari manis dan kelingking ditekuk).
    """
    # y-koordinat lebih kecil berarti lebih tinggi (karena titik 0,0 ada di kiri atas)
    
    # Jari telunjuk (Index) berdiri: ujung (8) lebih tinggi dari sendi bawahnya (6)
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    
    # Jari tengah (Middle) berdiri: ujung (12) lebih tinggi dari sendi bawahnya (10)
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    
    # Jari manis (Ring) ditekuk: ujung (16) lebih rendah dari sendi bawahnya (14)
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    
    # Jari kelingking (Pinky) ditekuk: ujung (20) lebih rendah dari sendi bawahnya (18)
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    return index_up and middle_up and ring_down and pinky_down

print("Tekan 'q' untuk keluar dari program.")

blur_transition_duration = 0.5  # durasi transisi dalam detik
current_blur_level = 0.0        # nilai 0.0 (tidak blur) hingga 1.0 (blur maksimal)
last_time = time.time()
max_blur_kernel = 99            # maksimal ukuran kernel blur

while True:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    success, img = cap.read()
    if not success:
        print("Gagal membaca kamera.")
        break
        
    # Flip gambar agar seperti cermin
    img = cv2.flip(img, 1)
    
    # Konversi BGR ke RGB (karena MediaPipe menggunakan RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Proses deteksi tangan
    results = hands.process(img_rgb)
    
    peace_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar titik dan garis pada tangan (opsional, bisa dihapus/dikomen jika tidak mau tampil)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Cek apakah pose jari adalah 'peace'
            if is_peace_sign(hand_landmarks):
                peace_detected = True
                
    # Menghitung level transisi blur
    if peace_detected:
        current_blur_level += dt / blur_transition_duration
    else:
        current_blur_level -= dt / blur_transition_duration
        
    # Membatasi nilai dari 0.0 hingga 1.0
    current_blur_level = max(0.0, min(1.0, current_blur_level))
    
    # Menerapkan efek blur
    if current_blur_level > 0.0:
        kernel_size = int(current_blur_level * max_blur_kernel)
        
        # Kernel size OpenCV harus bilangan ganjil
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, kernel_size)
        
        if kernel_size > 1:
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Menambahkan teks sebagai indikator jika blur sudah terlihat
        if current_blur_level > 0.2:
            cv2.putText(img, "Foto Kita Blur", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 2)
        
    cv2.imshow("Kamera Kita", img)
    
    # Tekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()
