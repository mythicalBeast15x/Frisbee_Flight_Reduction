import cv2

cap = cv2.VideoCapture("frisbee_videos/gopro_videos/gopro_frisbee_clip_8.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80)


while True:
    ret, frame = cap.read()
    #roi = frame[40:300, 300:640] # ---
    #print(frame.shape)
    if not ret:
        break
    # initial mask
    #mask = object_detector.apply(roi) #---
    mask = object_detector.apply(frame)
    #cv2.imshow('Frame', frame) # original frame
    cv2.imshow("Mask", mask)

    #refining mask
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #cv2.imshow("Contour", frame)
    #cv2.imshow("Contour Mask", mask) #refined mask

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()