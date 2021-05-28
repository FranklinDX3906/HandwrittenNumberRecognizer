# coding:utf-8
# 将图像进行分割


import cv2
import numpy as np
import torch
# from myCNN import MyCNN


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
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
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

    # borders.sort(key=sortlambda)
    # print(borders[0][0][0])
    return borders


# 显示结果及边框
def draw_borders(img, borders, results):
    # 绘制
    # print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        cv2.putText(img, str(results[i]), border[0],
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    return img


# 转化为28*28手写数据
def minist_img(img, borders, size=(28, 28)):
    img_data = np.zeros((len(borders), size[0], size[0]), dtype='uint8')
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = accessBinary(img)
    for i, border in enumerate(borders):
        border_img = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 扩展，否则图片有问题
        stride_length = (border_img.shape[0]-border_img.shape[1]) // 2
        if(stride_length > 0):
            border_img = cv2.copyMakeBorder(
                border_img, 7, 7, stride_length + 7, stride_length + 7, cv2.BORDER_CONSTANT)
        else:
            stride_length = -1*stride_length
            border_img = cv2.copyMakeBorder(
                border_img, stride_length+7, stride_length+7, 7, 7, cv2.BORDER_CONSTANT)

        border_img = cv2.resize(border_img, size)
        # border_img = np.expand_dims(border_img, axis=0)  # 扩维，否则不是图片
        img_data[i] = border_img
        # img_data[i] = np.expand_dims
    return img_data


# 预测数字
def prediction(img_data):
    # model = MyCNN()
    model = torch.load("X:/projects/HandwrittenNumberRecognizer/mycnn_minist.pth")
    # model =  torch.load(model_path)

    result = []

    for img in img_data:
        img = np.expand_dims(img, axis=0)  # 扩维，输入数据
        img = np.expand_dims(img, axis=0)  # 扩维，输入数据

        img = torch.Tensor(img)
        img = img.to('cuda:0')
        # print(img.size())
        output = model.forward(img)
        _, number = torch.max(output.data, 1)
        result.append(number.item())

    return result


# if __name__ == '__main__':

#     img_path = './test1.jpg'
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (1200, 900))  # 分辨率降低，否则太大

#     _img = grayscale_image(img)  # 灰度化
#     _img = inverse_img(_img)  # 反相
#     _img = binarization_img(_img)  # 二值化
#     # cv2.imshow('1',_img)
#     # cv2.waitKey(0)
#     borders = borders_img(_img)  # 找边框
#     # print(borders)
#     # borders_img = draw_borders(img, borders)  # 将边框放在图上
#     # cv2.imshow('test', borders_img)
#     # cv2.waitKey(0)
#     img_data = minist_img(_img, borders)
#     # cv2.imshow('test', img_data[0])
#     # print(imgdata[0])
#     # cv2.waitKey(0)
#     # model_path = './mycnn_minist.pth'
#     result = prediction(img_data)

#     # print(result)
#     img = draw_borders(img, borders, result)
#     cv2.imshow('test', img)
#     cv2.waitKey(0)
