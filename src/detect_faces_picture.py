# import opencv
import cv2 

def main():
    detect_faces_picture("assets/children.png", 1)

def detect_faces_picture(file, number):
    # implement object detection technique for face detection
    # initialize detector with pre-trained model parameters
    cc = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

    image = cv2.imread(file)

    # classifier trained to work with black and white images
    # convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # discretize grayscale image to turn image into black and white
    blackwhite_image = cv2.equalizeHist(grayscale_image)

    # detect faces
    faces = cc.detectMultiScale(blackwhite_image, scaleFactor = 1.3, minNeighbors = 4, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    # put a box around the faces
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("outputs/faces_detected_" + str(number) + ".png", image)
    
if __name__ == '__main__':
    main()