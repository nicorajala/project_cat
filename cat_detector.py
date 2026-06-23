import sys
import subprocess
import argparse
import cv2
import pyttsx3

from ultralytics import YOLO

cap = cv2.VideoCapture("http://192.168.1.59:5000/video")
cap.set(3, 640)                     # x res
cap.set(4, 480)                     # y res

model = YOLO("yolo/yolov8n.pt")

started = False

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    global started

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    global last_spoken

    while True:
        ret, img = cap.read()
        results = model(img, imgsz=320, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if(model.names[cls] == "cat" or cv2.waitKey(1) == ord('e')) and not started:
                started = True
                subprocess.Popen([sys.executable, '/home/nico/Programming/project_cat/main.py'])

        if args.no_gui:
            pass
        else:
            cv2.imshow('Dis shit', img)
            if cv2.waitKey(1) == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()