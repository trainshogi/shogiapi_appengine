import numpy as np
import cv2
import copy
from PIL import Image

debug = True

def makemochipic(img_org, points):
    # 先に塗りつぶしてある前提で動く
    img = copy.copy(img_org)
    points = np.array(points)
    img = cv2.fillPoly(img, pts=[points], color=(0,0,0))
    if debug:
        cv2.imwrite("LennaG1.png",img)
    # 変数とかいろいろ計算
    p_s = points[:,0]+points[:,1]
    minx = min(points[:,0])
    maxx = max(points[:,1])
    width = int((maxx-minx)/9)
    lt = points[np.argmin(p_s)]
    rb = points[np.argmax(p_s)]
    outw = int(3.5*width)
    insw = int(0.5*width)
    allw = outw+insw
    ltps = [0,0,allw,allw] # 左上x, 左上y, xwidth, ywidth
    rbps = [img.shape[1]-allw,img.shape[0]-allw,allw,allw] # 左上x, 左上y, xwidth, ywidth
    # 左上から
    if lt[0]>outw:
        ltps[0] = lt[0]-outw
    if lt[1]>insw:
        ltps[1] = lt[1]-insw
    if (img.shape[1]-rb[0])>outw:
        rbps[0] = rb[0]-insw
    if (img.shape[0]-rb[1])>insw:
        rbps[1] = rb[1]-outw
    # 切り出し
    gotemochi = img[ltps[1]:ltps[1]+allw,ltps[0]:ltps[0]+allw]
    sentemochi = img[rbps[1]:rbps[1]+allw,rbps[0]:rbps[0]+allw]
    if debug:
        cv2.imwrite("LennaG2.png",gotemochi)
        cv2.imwrite("LennaG3.png",sentemochi)
    gotemochi = cv2.flip(gotemochi, -1)
    return gotemochi, sentemochi

# def cv2pil(image):
#     ''' OpenCV型 -> PIL型 '''
#     new_image = image.copy()
#     if new_image.ndim == 2:  # モノクロ
#         pass
#     elif new_image.shape[2] == 3:  # カラー
#         new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     elif new_image.shape[2] == 4:  # 透過
#         new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
#     new_image = Image.fromarray(new_image)
#     return new_image

# def detect_mochigoma(gotemochi, sentemochi):
#     gote = detect_img(YOLO(),cv2pil(gotemochi))
#     sente = detect_img(YOLO(),cv2pil(sentemochi))    
#     return gote, sente

# def detect_img(yolo,img):
#     result = []
#     out_boxes, out_classes, out_scores, class_names = yolo.detect_image(img)
#     for i in range(len(out_boxes)):
#         print(str(i)+": "+class_names[out_classes[i]])
#         print(str(i)+": "+str(out_boxes[i]))
#         print(str(i)+": "+str(out_scores[i]))
#         if result == []:
#             result.append([out_classes[i],1])
#         elif result[-1][0] == out_classes[i]:
#             result[-1][1] += 1
#         else:
#             result.append([out_classes[i],1])
#     result.reverse()
#     yolo.close_session()
#     K.clear_session()
#     return result