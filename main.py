import cv2
cap = cv2.VideoCapture(2)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_roi = frame[y:y+h, x:x+h]
        gray_roi = gray[y:y+h, x:x+h]

        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)

        for xs, ys, ws, hs in smile:
            cv2.rectangle(face_roi, (xs, ys), (xs+ws, ys+hs), (0, 255, 255), 2)

    cv2.imshow('Selfie', frame)
    if cv2.waitKey(10) == ord('q'):
        break
