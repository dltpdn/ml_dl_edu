import numpy as np
import argparse
import time
import cv2

NET = 'SQZ'
#NET = 'GoogLe'

img_path = './images/barbershop.png'
label_path = "./synset_words.txt"

model_sqz = './models/squeezenet_v1.0.caffemodel'
confing_sqz = './models/squeezenet_v1.0.prototxt'
model_googLe = './models/bvlc_googlenet.caffemodel'
config_googLe = './models/bvlc_googlenet.prototxt'

if NET == 'SQZ':
    model = model_sqz
    config = confing_sqz
elif NET == 'GoogLe':
    model = model_googLe
    config = config_googLe

rows = open(label_path).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(config, model)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)      # 프레임 폭을 320으로 설정 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)     # 프레임 높이를 240으로 설정
while cap.isOpened():
    ret, image = cap.read()

    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))


    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    preds = preds.reshape((1, len(classes)))
    idx = np.argsort(preds[0])[::-1][0]
    label = classes[idx]
    proba = preds[0][idx]

    text = "Label: {}, {:.2f}%".format(label, proba * 100)
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2)
    print("[INFO]label: {}, probability: {:.5}".format(label, proba))

    # display the output image
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()