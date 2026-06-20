import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def processImage(img):
    img = cv2.erode(img, (5,5), iterations = 1)
    img = cv2.dilate(img, (5,5), iterations = 2)

    return img

def getCatMask(img):
    # I'm blue dabadedabadie
    lowerBound = np.array([100, 150, 50])
    upperBound = np.array([130, 255, 255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return processImage(cv2.inRange(hsv, lowerBound, upperBound))

def getBlobInfo(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])  # center x
    cy = int(M["m01"] / M["m00"])  # center y

    if area < 300:
        return None
    return cx, cy, area

def main():
    while True:
        ret, img = cap.read()
        mask = getCatMask(img)

        blob = getBlobInfo(mask)
        if blob is None:
            continue
        cx, cy, area = blob

        print(f"cx: {cx} cy: {cy} area: {area}")

        cv2.imshow('Dis shit', mask)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()