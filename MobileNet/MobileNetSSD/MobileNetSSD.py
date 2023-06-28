# import cv2 as cv
#
# # 模型路径
# model_bin = "D:/code/ai/ssd/MobileNetSSD_deploy.caffemodel"
# config_text = "D:/code/ai/ssd/MobileNetSSD_deploy.prototxt"
# # 类别信息
# objName = ["background",
#            "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair",
#            "cow", "diningtable", "dog", "horse",
#            "motorbike", "person", "pottedplant",
#            "sheep", "sofa", "train", "tvmonitor"]
#
# # 加载模型
# net = cv.dnn.readNetFromCaffe(config_text, model_bin)
#
# # 获得所有层名称与索引
# layerNames = net.getLayerNames()
# lastLayerId = net.getLayerId(layerNames[-1])
# lastLayer = net.getLayer(lastLayerId)
# print(lastLayer.type)
#
# # 打开摄像头
# cap = cv.VideoCapture("D:/PyCharm 2022.2.4/pythonProject/scientificProject/train/tv.mp4")
# # cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if ret is False:
#         break
#     h, w = frame.shape[:2]
#     blobImage = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)
#     net.setInput(blobImage)
#     cvOut = net.forward()
#     for detection in cvOut[0, 0, :, :]:
#         score = float(detection[2])
#         objIndex = int(detection[1])
#         if score > 0.5:
#             left = detection[3] * w
#             top = detection[4] * h
#             right = detection[5] * w
#             bottom = detection[6] * h
#
#             # 绘制
#             cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
#             cv.putText(frame, "score:%.2f, %s" % (score, objName[objIndex]),
#                        (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8)
#     # 显示
#     cv.imshow('video-ssd-demo', frame)
#     c = cv.waitKey(10)
#     if c == 27:
#         break
#
# cv.waitKey(0)
# cv.destroyAllWindows()

import os
import cv2
# import cvzone
import numpy as np

# 设置图片的宽度和高度
img_width, img_heigth = 300, 300
# 得到图像的高宽比
WHRatio = img_width / float(img_heigth)
# 设置图片的缩放因子
ScaleFactor = 0.007843
# 设置平均数
meanVal = 127.5
# 设置置信度阈值
threshod = 0.2

# mobileNetSSD可以检测类别数21=20+1（背景）
classNames = ['background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']

# 加载文件
net = cv2.dnn.readNetFromCaffe(prototxt='D:/code/ai/ssd/MobileNetSSD_deploy.prototxt',
                               caffeModel='D:/code/ai/ssd/MobileNetSSD_deploy.caffemodel')


# 对图片进行处理和设置网络的输入同时进行前向传播
def processImage(imgSize):
    # 对图片进行预处理
    blob = cv2.dnn.blobFromImage(image=imgSize, scalefactor=ScaleFactor,
                                 size=(img_width, img_heigth), mean=meanVal)
    # 设置网络的输入并进行前向传播
    net.setInput(blob)
    detections = net.forward()
    # 对图像进行按比例裁剪
    height, width, channel = np.shape(imgSize)
    if width / float(height) > WHRatio:
        # 裁剪多余的宽度
        cropSize = (int(height * WHRatio), height)
    else:
        # 裁剪多余的高度
        cropSize = (width, int(width / WHRatio))
    y1 = int((height - cropSize[1]) / 2)
    y2 = int(y1 + cropSize[1])
    x1 = int((width - cropSize[0]) / 2)
    x2 = int(x1 + cropSize[0])
    imgSize = imgSize[y1:y2, x1:x2]
    height, width, channel = np.shape(imgSize)

    # 遍历检测的目标
    # print('detection.shape: {}'.format(detections.shape))
    # print('detection: {}'.format(detections))
    for i in range(detections.shape[2]):
        # 保留两位小数
        confidence = round(detections[0, 0, i, 2] * 100, 2)
        if confidence > threshod:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * width)
            yLeftBottom = int(detections[0, 0, i, 4] * height)
            xRightTop = int(detections[0, 0, i, 5] * width)
            yRightTop = int(detections[0, 0, i, 6] * height)

            cv2.rectangle(img=imgSize, pt1=(xLeftBottom, yLeftBottom),
                          pt2=(xRightTop, yRightTop), color=(0, 255, 0), thickness=2)
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # cvzone.putTextRect(img=imgSize, text=label, pos=(xLeftBottom + 9, yLeftBottom - 12),
            #                    scale=1, thickness=1, colorR=(0, 255, 0))
            cv2.rectangle(imgSize, (xLeftBottom, yLeftBottom - labelSize[1]),
                          (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(imgSize, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return imgSize


# 对单张图片进行检测
def SignalDetect(img_path='../../train/pictures/p10.jpg'):
    imgSize = cv2.imread(img_path)
    imgSize = processImage(imgSize)
    cv2.imshow('imgSize', imgSize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 实时检测
def detectTime():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(src=frame, dsize=(520, 520))
        frame = cv2.flip(src=frame, flipCode=2)
        frame = processImage(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Pycharm')
    SignalDetect()
    # detectTime()
