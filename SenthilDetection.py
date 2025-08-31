import cv2
import numpy as np

net = cv2.dnn.readNet("SenthilDetection_dnn_model/yolov3_training_last_SenthilFacePose.weights",
                      "SenthilDetection_dnn_model/yolov3-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []
with open("SenthilDetection_dnn_model/COCODataset.txt", "r") as file_object:
    for class_name in file_object.readlines():
        classes.append(class_name.strip())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

button_person = False

def click_button (event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
    polygon = np.array([[(20,20), (220,20), (220,70), (20,70)]])
    is_inside = cv2.pointPolygonTest(polygon, (x,y), False)
    if is_inside > 0:
        print ("We are clicking inside the button")

        if button_person is False:
            button_person = True
        else:
            button_person = False

cv2.namedWindow("Object Detection")
cv2.setMouseCallback("Object Detection", click_button)

while cap.isOpened():

    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        if score > 0.9:
            x, y, w, h = bbox
            class_name = classes[class_id]

            print ("Class ID is " + str(class_id));

            print ("Score of identified object is " + str(score));

            if class_name == "senthil": #and button_person is True:
                cv2.putText(frame, class_name, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,256,0), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,256,0), 3)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
