import cv2
from deepface import DeepFace
import pygame
import time
import numpy as np

pygame.mixer.init()


#sound dictionary
emotion_sounds = {
    "happy": "sounds/happy.wav",
    "sad": "sounds/sad.wav",
    "angry": "sounds/angry.wav",
    "surprise": "sounds/surprise.wav",
    "neutral": "sounds/neutral.wav",
    "fear": "sounds/sad.wav",
    "disgust": "sounds/angry.wav"
}

#emoji dictionary
emoji_images = {
    "happy": cv2.imread("emojis/happy.png", cv2.IMREAD_UNCHANGED),
    "sad": cv2.imread("emojis/sad.png", cv2.IMREAD_UNCHANGED),
    "angry": cv2.imread("emojis/angry.png", cv2.IMREAD_UNCHANGED),
    "fear": cv2.imread("emojis/fear.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("emojis/surprise.png", cv2.IMREAD_UNCHANGED),
    "neutral": cv2.imread("emojis/neutral.png", cv2.IMREAD_UNCHANGED),
}

for key, img in emoji_images.items():
    if img is None:
        print("❌ NOT LOADED:", key)
    else:
        print("✅ LOADED:", key)


#resizing image
def overlay_emoji(frame, emoji, x, y):
    h, w = emoji.shape[:2]

    if emoji.shape[2] == 4:  # Has alpha
        alpha = emoji[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]

        emoji_rgb = emoji[:, :, :3]
        frame_section = frame[y:y+h, x:x+w]

        blended = (alpha * emoji_rgb + (1 - alpha) * frame_section).astype(np.uint8)
        frame[y:y+h, x:x+w] = blended

    else:
        frame[y:y+h, x:x+w] = emoji



last_emotion = None
last_play_time = 0
cooldown = 3

cap = cv2.VideoCapture(0)

frame_count = 0
emotion = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    #emotion detection
    if frame_count % 10 == 0:
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False
            )
            emotion = result[0]['dominant_emotion'].lower()
            print("Detected emotion:", emotion)
        except:
            emotion = "Detecting..."

    # sound logic
    current_time = time.time()

    if emotion != last_emotion and (current_time - last_play_time) > cooldown:
        if emotion in emotion_sounds:
            try:
                pygame.mixer.music.load(emotion_sounds[emotion])
                pygame.mixer.music.play()
                last_play_time = current_time
                last_emotion = emotion
                print("Sound played for:", emotion)
            except Exception as e:
                print("Sound error:", e)

    #emotion logic
    if emotion in emoji_images and emoji_images[emotion] is not None:
                emoji = emoji_images[emotion]

                h, w, _ = emoji.shape
                x, y = 50, 100   # position on screen

                # Check frame boundaries
                if y + h < frame.shape[0] and x + w < frame.shape[1]:

                    if emoji.shape[2] == 4:  # PNG with alpha
                        alpha = emoji[:, :, 3] / 255.0
                        for c in range(3):
                            frame[y:y+h, x:x+w, c] = (
                                alpha * emoji[:, :, c] +
                                (1 - alpha) * frame[y:y+h, x:x+w, c]
                            )
                    else:
                        frame[y:y+h, x:x+w] = emoji

    display_text = emotion

    cv2.putText(frame, display_text, (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2)
    
    emoji = emoji_images.get(emotion)

    if emoji is not None:
        emoji = cv2.resize(emoji, (120,120))
        overlay_emoji(frame, emoji, 50, 100)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
