import screen_brightness_control as sbc
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def koordinat_getir(landmarks, indeks, h, w):
    landmark = landmarks[indeks]
    return int(landmark.x * w), int(landmark.y * h)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    h, w, c = annotated_image.shape

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)  # işaret parmak ucu
        x2, y2 = koordinat_getir(hand_landmarks, 4, h, w)  # baş parmak ucu
        renk = (255, 255, 0)

        annotated_image = cv2.circle(annotated_image, (x1, y1), 9, renk, 5)
        annotated_image = cv2.circle(annotated_image, (x2, y2), 9, renk, 5)
        annotated_image = cv2.line(annotated_image, (x1, y1), (x2, y2), renk, 5)

        uzaklik = int(np.hypot(x2 - x1, y2 - y1))  # parmaklar arasındaki mesafe
        xort = (x1 + x2) // 2
        yort = (y1 + y2) // 2
        annotated_image = cv2.circle(annotated_image, (xort, yort), 9, (0, 255, 255), 5)
        annotated_image = cv2.putText(annotated_image, str(uzaklik), (xort, yort),
                                      cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)

        handedness = handedness_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Başparmak ve işaret parmağı arasındaki mesafeye bağlı olarak parlaklık ayarını yapma
        brightness = calculate_brightness(uzaklik)
        sbc.set_brightness(brightness)  # Ekran parlaklığı ayarlama

    return annotated_image

def calculate_brightness(distance):
    min_distance = 20  # Minimum mesafe
    max_distance = 200  # Maksimum mesafe

    # Mesafeyi, 0 ile 100 arasında bir parlaklık değerine dönüştürme
    brightness = max(0, min(100, (distance - min_distance) / (max_distance - min_distance) * 100))
    return int(brightness)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# kamera açık olduğu sürece
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)  # 1 ms bekle
        if key == ord('q') or key == ord('Q'): 
            break

        # Parlaklığın çok değişkenlik göstermemesi için
last_brightness = -1

def update_brightness_safely(brightness):
    global last_brightness
    if abs(brightness - last_brightness) >= 5:  # %5'ten fazla değişirse güncelle
        update_brightness_safely(brightness)
        last_brightness = brightness


cam.release()
cv2.destroyAllWindows()