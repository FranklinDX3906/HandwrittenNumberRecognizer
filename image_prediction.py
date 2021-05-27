# coding:utf-8
# 将图像进行分割


import cv2
import numpy as np


# 将图片灰度化
def grayscale_image(img):
    # img = cv2.resize(img, (300, 400))

    height = img.shape[0]  # 高度
    width = img.shape[1]  # 宽度
    gray_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            gray_img[i, j] = max(img[i, j][1], img[i, j][1], img[i, j][1])

    # cv2.imshow('1', gray_img)
    return gray_img


# 反相灰度图，将黑白阈值颠倒
def inverse_img(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i][j] = 255 - img[i][j]
    # cv2.imshow('2', img)
    return img


# 二值化图像
def binarization_img(img, threshold=128):
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    # cv2.imshow('3', img)
    return img


# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def borders_img(img, minsize=50, maxsize=5000):
    # img = accessBinary(img)  # 二值化
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        size = w*h
        if (size > minsize) & (size < maxsize):
            border = [(x, y), (x+w, y+h)]
            borders.append(border)
    return borders


# 显示结果及边框
def draw_borders(img, borders):
    # 绘制
    # print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    return img


if __name__ == '__main__':

    img_path = './test4.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1200, 900))  # 分辨率降低，否则太大

    _img = grayscale_image(img)  # 灰度化
    _img = inverse_img(_img)  # 反相
    _img = binarization_img(_img)  # 二值化
    borders = borders_img(_img)  # 找边框
    _img = draw_borders(img, borders)
    cv2.imshow('test', _img)
    cv2.waitKey(0)
