import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "/Users/anushkajha/Documents/GitHub/ASL-Converter/assets/yes"

while True: 
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue
    
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrp = img[max(0, y - offset): y + h + offset, max(0, x - offset): x + w + offset]
        
        imgCrpShape = imgCrp.shape
        aspectRatio = h / w

        # Resize based on aspect ratio
        if aspectRatio > 1:  # Height is greater than width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrp, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            
            if wCal <= imgSize:  # Check if the width fits
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                print("Resized image width exceeds white image width.")

        else:  # Width is greater than height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrp, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            
            if hCal <= imgSize:  # Check if the height fits
                imgWhite[hGap:hGap + hCal, :] = imgResize
            else:
                print("Resized image height exceeds white image height.")

        # Display images
        cv2.imshow('ImageCrop', imgCrp)
        cv2.imshow('ImageWhite', imgWhite)

    # Show the original image
    cv2.imshow('Original Image', img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved Image {counter}")

cap.release()
cv2.destroyAllWindows()
