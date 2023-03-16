#Age and gender of a person determines the type of food item they are likely to consume. Predicting the age and gender of a person can help in ordering of product to some extent. For example if the customer is a girl, she is more likely to order some chocolate stuff, if the customer is a faculty they are more likely to order tea or snacks like samosa. Offering chinese snacks can attracts young people aging between 18-25.
#The task of detecting age and gender, however, is an inherently difficult problem, more so than many other computer vision tasks.
#The reason is that to have tags for such images, we need to access the personal information of the subjects in the images.
#Prediction of ages can only start by writing the code for detecting faces because without face detection we will not be able to move further with the task of age and gender prediction.
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes
  #
  genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageNet = cv.dnn.readNet(ageModel, ageProto)

genderList = ['Male', 'Female']

blob = cv.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
genderNet.setInput(blob)
genderPreds = genderNet.forward()
gender = genderList[genderPreds[0].argmax()]
print("Gender Output : {}".format(genderPreds))
print("Gender : {}".format(gender))
#
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv.dnn.readNet(ageModel, ageProto)

ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

ageNet.setInput(blob)
agePreds = ageNet.forward()
age = ageList[agePreds[0].argmax()]
print("Gender Output : {}".format(agePreds))
print("Gender : {}".format(age))
#
label = "{}, {}".format(gender, age)
cv.putText(frameFace, label, (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3, cv.LINE_AA)
cv.imshow("Age Gender Demo", frameFace)
