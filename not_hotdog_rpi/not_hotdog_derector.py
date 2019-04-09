import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from imutils.video import VideoStream
import time


MODEL_PATH = 'hotdog_not_hotdog.model'
TOTAL_CONSEC = 0
TOTAL_THRESH = 20
# initialize is the santa alarm has been triggered
SANTA = False

# load the model
model = load_model(MODEL_PATH)
print("[INFO] model loaded..")

#cap = cv2.VideoCapture(0)
vs = VideoStream().start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
print('[INFO] video capture is ready..')
#while cap.isOpened():
while True:
    # prepare the image to be classified by our deep learning network
    #ret, frame = cap.read()
    frame = vs.read()
    #if not ret:
    #    print("no frame from capture.")
    #    break
    image = cv2.resize(frame, (28, 28))
    print(image.shape)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    print(image.shape)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
 
    # classify the input image and initialize the label and
    # probability of the prediction
    (notSanta, santa) = model.predict(image)[0]
    label = "Not Hotdog"
    proba = notSanta
    # check to see if santa was detected using our convolutional
    # neural network
    if santa > notSanta:
        # update the label and prediction probability
        label = "Hotdog"
        proba = santa
        color = (0,255,0)
        # increment the total number of consecutive frames that
        # contain santa
        TOTAL_CONSEC += 1
        # check to see if we should raise the santa alarm
    else:
        TOTAL_CONSEC = 0
        SANTA = False
        color = (0,0,255)
    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q") or key == 27:
        break
 
# do a bit of cleanup
print("[INFO] cleaning up...")
#cap.release()
vs.stop()
cv2.destroyAllWindows()