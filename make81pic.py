# -*- coding: utf-8 -*-
from PIL import Image,ImageOps
# from corner import find_corner

import cv2,copy,time
import numpy as np
# from keras.preprocessing.image import array_to_img, img_to_array
import glob
import os, re
from datetime import datetime
from operator import itemgetter
from corner import find_corner
from makemochipic import makemochipic

release = True

def transform_by4(img, points):
    """ 4点を指定してトリミングする。 """
    points = sorted(points, key=lambda x: x[1])  # yが小さいもの順に並び替え。
    top = sorted(points[:2], key=lambda x: x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
    bottom = sorted(points[2:], key=lambda x: x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
    points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。

    width = max(np.sqrt(((points[0][0] - points[2][0]) ** 2) * 2),
                np.sqrt(((points[1][0] - points[3][0]) ** 2) * 2))
    height = max(np.sqrt(((points[0][1] - points[2][1]) ** 2) * 2),
                 np.sqrt(((points[1][1] - points[3][1]) ** 2) * 2))

    dst = np.array([
        np.array([0, 0]),
        np.array([width - 1, 0]),
        np.array([width - 1, height - 1]),
        np.array([0, height - 1]),
    ], np.float32)

    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。

def make81pic(im, rotate):
    print("come to make81pic.")
    image_save = True
    image_save_one = False
    size = 64

    if release:
        # ファイル形式をOpenCV用に変換
        im = np.fromstring(im, np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    
    # 時間を取得
    if image_save:
        nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs('../reco/' + nowtime)
        cv2.imwrite('../reco/' + nowtime + '/' + 'ban0.jpg', im)
        # cv2.imwrite('ban1.jpg',im)

    # 辞書→リスト
    # points = list(trapezoid.values())
    # points = [[i['x'],i['y']] for i in points]
    # 角を探す
    print("come to find corner.")

    points = find_corner(im)    

    print("finish to find corner.")

    # 持ち駒画像作成
    gotemochi, sentemochi= makemochipic(im, points)

    # トリミング
    # im = transform_by4(im, [[1005,309],[3176,315],[1055,2616],[3099,2642]])
    im = transform_by4(im, points)
    # 画像にだしておく
    if image_save:
        cv2.imwrite('../reco/' + nowtime + '/' + 'ban2.png', im)
        cv2.imwrite('../reco/' + nowtime + '/' + 'gote.png', gotemochi)
        cv2.imwrite('../reco/' + nowtime + '/' + 'sente.png', sentemochi)
    im = cv2.resize(im,(size * 9, size * 9))
    img_list = []
    # 一枠ずつにする
    for i in range(9):
        for j in range(9):
            # ただ分割して保存
            img = im[j * size:(j + 1) * size, i * size:(i + 1) * size]
            if image_save_one:
                cv2.imwrite('reco/' + nowtime +'/-' + str(j) + str(i) + '.png',img)
            # OpenCVとKerasだとBlueとRedが変わってる問題
            b,g,r=cv2.split(img)
            img=cv2.merge((r,g,b))
            img_list.append(img)
    print("end make81pic.")
    return img_list, gotemochi, sentemochi # im, gotemochi, sentemochi
