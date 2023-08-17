# import opencv
import cv2 

def main():
    detect_faces_camera()

def detect_faces_camera():
    # implement object detection technique for face detection
    # initialize detector with pre-trained model parameters
    cc = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    
    # constant capturing requires infinite loop
    while(True):
        # start video capture
        video_capture = cv2.VideoCapture(0)

        # ~ VideoCapture() returns retval (probably bool type), image

        retval, frame = video_capture.read()
        # classifier trained to work with black and white images
        # convert image to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # discretize grayscale image to turn image into black and white
        blackwhite_frame = cv2.equalizeHist(grayscale_frame)

        # detect faces
        faces = cc.detectMultiScale(blackwhite_frame, scaleFactor = 1.5, minNeighbors = 4, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

        # put a box around the faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show rectangle on live video feed
        cv2.imshow('frame', frame) # ~ params: ("name of frame", frame)

        # if user input is q (for quit), stop video capture
        # cv2.waitKey(x) returns a 32-bit int corresponding to pressed key or -1 if none after x milliseconds

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    video_capture.release() # releases captured frame
    cv2.destroyAllWindows() # close windows

if __name__ == '__main__':
    main()