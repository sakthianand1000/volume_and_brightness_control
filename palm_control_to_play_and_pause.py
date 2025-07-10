import cv2
import mediapipe as mp
import math
import screen_brightness_control as sbc
import numpy as np
import time
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Set up camera
wCam, hCam = 740, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize variables
pTime = 0
bright = 0
bBar = 400
vol = 0
vBar = 400
is_playing = True  # Flag to track play/pause state
gesture_detected = False  # Flag to track if the gesture is currently active

# Initialize volume controller
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

while True:
    # Read camera frame
    success, img = cap.read()

    # Find hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw hands and find landmarks
    if results.multi_hand_landmarks:
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            handType = handedness.classification[0].label

            # Draw hand
            h, w, c = img.shape
            cv2.rectangle(img, (20, 20), (w - 20, h - 20), (0, 255, 0), 2)

            # Find fingers
            fingers = []
            for id, lm in enumerate(handLms.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                fingers.append((x, y))

            # Control brightness (right hand)
            if handType == 'Right':
                # Find distance between thumb and index finger
                x1, y1 = fingers[4][0], fingers[4][1]
                x2, y2 = fingers[8][0], fingers[8][1]
                length = math.hypot(x2 - x1, y2 - y1)

                # Update brightness bar
                bBar = np.interp(length, [50, 250], [400, 150])
                bright = np.interp(length, [50, 250], [0, 100])
                sbc.set_brightness(int(bright))

                # Draw brightness bar
                cv2.rectangle(img, (50, 150), (75, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(bBar)), (75, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'Brightness: {int(bright)}%', (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Control volume (left hand)
            elif handType == 'Left':
                # Find distance between thumb and index finger
                x1, y1 = fingers[4][0], fingers[4][1]
                x2, y2 = fingers[8][0], fingers[8][1]
                length = math.hypot(x2 - x1, y2 - y1)

                # Update volume bar
                vBar = np.interp(length, [50, 250], [400, 150])
                vol = np.interp(length, [50, 250], [0, 100])
                volume.SetMasterVolumeLevelScalar(vol / 100, None)

                # Draw volume bar
                cv2.rectangle(img, (w - 125, 150), (w - 100, 400), (0, 0, 255), 3)
                cv2.rectangle(img, (w - 125, int(vBar)), (w - 100, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f'Volume: {int(vol)}%', (w - 200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # Detect palm to play/pause (both hands)
            if handType == 'Left' or handType == 'Right':
                # Get the distance between the thumb (id 4) and the pinky (id 20)
                thumb_x, thumb_y = fingers[4]
                pinky_x, pinky_y = fingers[20]
                distance = math.hypot(pinky_x - thumb_x, pinky_y - thumb_y)

                # Palm detection: If thumb and pinky are far apart, consider it as "palm open" gesture
                if distance > 150:  # Adjust this threshold value
                    if not gesture_detected:  # Only trigger once per gesture
                        if is_playing:
                            # Pause the video by simulating spacebar key press
                            pyautogui.press('space')  # Simulate spacebar press (pause)
                            is_playing = False
                            cv2.putText(img, 'Video Paused', (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
                        else:
                            # Play the video by simulating spacebar key press
                            pyautogui.press('space')  # Simulate spacebar press (play)
                            is_playing = True
                            cv2.putText(img, 'Video Playing', (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
                        gesture_detected = True  # Mark gesture as detected
                else:
                    gesture_detected = False  # Reset gesture detection when hand is closed

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with hand landmarks and control bars
    cv2.imshow("Hand Tracking", img)

    # Exit the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
