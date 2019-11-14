import RPi.GPIO as GPIO
import time
import csv

sw1, sw2 = 18, 23
led1, led2 =24 , 25

GPIO.setmode(GPIO.BCM)
GPIO.setup(sw1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(led1, GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(sw2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(led2, GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)

csv_file = open('gpio.csv', 'a')
gpio_writer = csv.writer(csv_file, delimiter=',')
gpio_writer.writerow(["in_1","in_2","out_1","out_2"])

in1, in2, out1, out2 = (0,0,0,0)
pre_in1, pre_in2, pre_out1, pre_out2 = (-1,-1,-1,-1)

try:
    while True:
        in1, in2, out1, out2 = GPIO.input(sw1), GPIO.input(sw2), GPIO.input(led1), GPIO.input(led2)
        if pre_in1 != in1 or pre_in2 != in2:
            pre_in1 = in1
            pre_in2 = in2
            pre_out1 = out1
            pre_out2 = out2
            print(in1, in2, out1, out2)
            gpio_writer.writerow([in1, in2, out1, out2])
            csv_file.flush()
            time.sleep(0.5)

finally:
    GPIO.cleanup()
    csv_file.close()    
