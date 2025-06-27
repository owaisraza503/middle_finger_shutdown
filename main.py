import cv2
import mediapipe as mp
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

middle_detected_time = 0
shutdown_delay = 3  # seconds

def is_middle_finger_up(hand_landmarks):
    # Landmarks for finger tips
    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # If only middle finger is up
    return fingers == [0, 1, 0, 0]

print("🚀 Running... Show your middle finger to shut down (or press Q to quit)")

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if is_middle_finger_up(handLms):
                print("🖕 Middle finger detected!")
                if time.time() - middle_detected_time > shutdown_delay:
                    os.system("shutdown now")
                else:
                    print("⏳ Hold for 3 seconds to confirm shutdown")
                middle_detected_time = time.time()

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Middle Finger Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
