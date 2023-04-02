import cv2
import numpy as np





min_width_rectangle = 30
min_height_rectangle = 30

count_line_position = 600

# Initialize Substructor
algo = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=50)


def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []
offset = 5  # Alowable error b/w pixel
counter = 0

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()



    # if frame is read correctly ret is True
    if ret == True:


        blur = cv2.GaussianBlur(frame, (3, 3), 5)
        vid_sub = algo.apply(blur)
        dilat = cv2.dilate(vid_sub, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_OPEN, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        counters, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame, (250, count_line_position), (2000, count_line_position), (255, 0, 0), 3)
        for (i, c) in enumerate(counters):
            (x, y, w, h) = cv2.boundingRect(c)
            val_counter = (w >= min_width_rectangle) and (h >= min_height_rectangle)

            if not val_counter:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Vehicle No: " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            for (x, y) in detect:
                if y < (count_line_position + offset) and y > (count_line_position - offset):
                    counter += 1
                cv2.line(frame, (250, count_line_position), (2000, count_line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Vehicle No: " + str(counter))










        cv2.putText(frame, "Vehicle No: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow('Detector', frame)

        if cv2.waitKey(2) == ord('q'):
            break

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


cap.release()
cv2.destroyAllWindows()










