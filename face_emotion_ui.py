import cv2
from deepface import DeepFace
from tkinter import *
from PIL import Image, ImageTk
import threading

def start_camera():
    def run_camera():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotion_label.config(text=f"Emotion: {emotion}")
            except Exception:
                emotion_label.config(text="Detecting...")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.update()

            if stop_flag.get():
                break

        cap.release()
        cv2.destroyAllWindows()

    stop_flag.set(False)
    thread = threading.Thread(target=run_camera)
    thread.start()

def stop_camera():
    stop_flag.set(True)
    emotion_label.config(text="Emotion: None")
    video_label.config(image='')

root = Tk()
root.title("AI Face Emotion Detector")
root.geometry("700x550")
root.config(bg="#f0f0f0")

title = Label(root, text="😊 Face Emotion Detector 😊", font=("Arial", 22, "bold"), bg="#f0f0f0", fg="#333")
title.pack(pady=10)

video_label = Label(root)
video_label.pack()

emotion_label = Label(root, text="Emotion: None", font=("Arial", 18), bg="#f0f0f0", fg="blue")
emotion_label.pack(pady=15)

button_frame = Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

start_btn = Button(button_frame, text="Start Camera", font=("Arial", 14), bg="green", fg="white", command=start_camera)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = Button(button_frame, text="Stop Camera", font=("Arial", 14), bg="red", fg="white", command=stop_camera)
stop_btn.grid(row=0, column=1, padx=10)

stop_flag = BooleanVar()
stop_flag.set(False)

root.mainloop()
