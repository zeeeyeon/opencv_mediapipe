from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def remove_background(image: np.ndarray) -> np.ndarray:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    fg_mask = bg_subtractor.apply(image)
    result = cv2.bitwise_and(image, image, mask=fg_mask)
    return result

def detect_wingspan(image: np.ndarray, height: float):
    image_no_bg = remove_background(image)
    rgb_image = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        return {"error": "사람이 감지되지 않았습니다. 전체 몸이 보이도록 조정하세요."}

    landmarks = results.pose_landmarks.landmark

    left_fingertip = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    right_fingertip = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
    head = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if not all([left_fingertip.visibility > 0.5, right_fingertip.visibility > 0.5]):
        return {"error": "손끝 키포인트가 감지되지 않았습니다. 팔을 완전히 벌려주세요."}

    if not all([head.visibility > 0.5, left_ankle.visibility > 0.5, right_ankle.visibility > 0.5]):
        return {"error": "신체 키포인트 감지가 충분하지 않습니다. 카메라 각도를 조정하세요."}

    img_h, img_w, _ = image.shape
    left_hand_x = left_fingertip.x * img_w
    left_hand_y = left_fingertip.y * img_h
    right_hand_x = right_fingertip.x * img_w
    right_hand_y = right_fingertip.y * img_h
    head_x = head.x * img_w
    head_y = head.y * img_h
    left_ankle_x = left_ankle.x * img_w
    left_ankle_y = left_ankle.y * img_h
    right_ankle_x = right_ankle.x * img_w
    right_ankle_y = right_ankle.y * img_h

    pixel_wingspan = np.linalg.norm([left_hand_x - right_hand_x, left_hand_y - right_hand_y])
    pixel_height = np.linalg.norm([head_x - (left_ankle_x + right_ankle_x) / 2, head_y - (left_ankle_y + right_ankle_y) / 2])
    real_wingspan = (pixel_wingspan / pixel_height) * height

    if real_wingspan < height * 0.9 or real_wingspan > height * 1.15:
        return {"error": f"윙스팬 계산 오류. 포즈나 카메라 각도를 조정하세요. (계산된 값: {real_wingspan:.2f}cm)"}

    return {
        "wingspan": round(real_wingspan, 2),
    }

@app.post("/api/user/wingspan")
async def calculate_wingspan(file: UploadFile = File(...), height: float = Form(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    result = detect_wingspan(image, height)

    if "error" in result:
        return JSONResponse(content=result, status_code=400)

    return result
