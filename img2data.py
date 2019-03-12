
# coding: utf-8

# In[6]:


import cv2
import numpy as np

def digit2data(src, size=(8,8), reshape=True):
    h, w = src.shape[:2]
    square = src
    # 정사각형 형태로 만들기
    if h > w:
        pad = (h - w)//2
        square = np.zeros((h, h), dtype=np.uint8)
        square[:, pad:pad+w] = src
    elif w > h :
        pad = (w - h)//2
        square = np.zeros((w, w), dtype=np.uint8)
        square[pad:pad+h, :] = src
    px = np.zeros(size, np.uint8)
    px[:, :] = cv2.resize(square, size, interpolation=cv2.INTER_AREA)
    if reshape:
        px = px.reshape((1,size[0] * size[1]))
    return px 

def img2digits(image, size=None, reshape=True):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hirachy = cv2.findContours(th, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
    numbers = []
    for c in contours:
        # 컨투어를 감싸는 외접 사각형으로 숫자 영역 좌표 구하기 ---⑦
        (x, y, w, h) = cv2.boundingRect(c)    
        # 외접 사각형의 크기가 너무 작은것은 제외 ---⑧
        if w >= 5 and h >= 25:
            # 숫자 영역만 roi로 확보하고 사각형 그리기 ---⑨
            roi = gray[y:y + h, x:x + w]
            numbers.append(roi)

    if size is None:
        return numbers
    else:
        for i, n in enumerate(numbers):
            n = 255- n  # 반전
            numbers[i] = digit2data(n, size, reshape)
        return numbers
