import pandas as pd
import RPi.GPIO as GPIO
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('gpio.csv')
classes = np.unique(df[['out_1', 'out_2']], axis=0)



def labeling(row):
    return np.where( np.all(classes==[row.out_1, row.out_2], axis=1))[0][0]

df['class'] = df.apply(labeling, axis=1)
x = np.array(df[['in_1', 'in_2']])
y = np.array(df[['class']]).ravel()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)


sw1, sw2 = 18, 23
led1, led2 =24 , 25

GPIO.setmode(GPIO.BCM)
GPIO.setup(sw1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(sw2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(led1, GPIO.OUT)
GPIO.setup(led2, GPIO.OUT)

prev_in1, prev_in2 = -1,-1
try:
    while True:
        in1, in2= GPIO.input(sw1), GPIO.input(sw2)
        if prev_in1 != in1 or prev_in2 != in2:
            out = knn.predict([[in1, in2]])
            out1, out2 = classes[out[0]]
            out1, out2 = out1.item(), out2.item()
            GPIO.output(led1, out1)
            GPIO.output(led2, out2)
            print(in1, in2, out1, out2)
            prev_in1, prev_in2 = in1, in2
            time.sleep(0.1)
finally:
    GPIO.cleanup()