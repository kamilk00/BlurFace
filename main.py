import cv2
import os 

dirPath = os.getcwd()

#detecting a face from an image
def face_detection(faceNet, frame):

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []

    for i in range(detection.shape[2]):

        confidence = detection[0, 0, i, 2]
        if confidence > 0.65:

            x1 = int(detection[0, 0, i, 3] * width)
            y1 = int(detection[0, 0, i, 4] * height)
            x2 = int(detection[0, 0, i, 5] * width)
            y2 = int(detection[0, 0, i, 6] * height)
            faceBoxes.append([x1, y1, x2, y2])


    return faceBoxes

video = cv2.VideoCapture(0)
padding = 15

#using pretrained model to detect faces
face_pbtxt = dirPath + "\opencv_face_detector.pbtxt"
face_pb = dirPath + "\opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(face_pb, face_pbtxt)

while cv2.waitKey(1) < 0:

    _, img = video.read()
    faceBoxes = face_detection(faceNet, img)

    for box in faceBoxes:

        face = img[max(0, box[1] - padding):min(box[3] + padding, img.shape[0] - 1), 
                   max(0, box[0] - padding):min(box[2] + padding, img.shape[1] - 1)]
        
        #using average filter to blur each face
        face = cv2.blur(face, (55, 55))

        img[max(0, box[1] - padding):min(box[3] + padding, img.shape[0] - 1), 
            max(0, box[0] - padding):min(box[2] + padding, img.shape[1] - 1)] = face

    cv2.imshow("FaceBlur", img)

video.release()
cv2.destroyAllWindows()