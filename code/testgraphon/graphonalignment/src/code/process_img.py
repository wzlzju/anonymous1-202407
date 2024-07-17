import cv2 as cv
import numpy as np
import os, sys


def process_epx3():
    path = "images/exp3/"
    res = "images/exp3 resize/"
    files = os.listdir(path)

    for file in files:
        img = cv.imread(path+file)
        img = img[85:525, 170:850]
        cv.imwrite(res+file, img)

def process_exp1():
    path = "images/exp1/"
    res = "images/exp1 resize/"

    img = cv.imread(path+"exp1.png")
    h, w, _ = img.shape
    r = 2
    c = 4
    sub_h = h//r
    sub_w = w//c-4
    for i in range(r):
        for j in range(c):
            sub_img = img[sub_h*i: sub_h*(i+1), (sub_w+j*j)*j: (sub_w+j*j)*(j+1), :]
            print(sub_img.shape)
            cv.imwrite(res+"exp 1-%d-%d.png" % (i,j), sub_img)



if __name__ == "__main__":
    process_exp1()