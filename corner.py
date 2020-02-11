import cv2,copy,math
import numpy as np
from operator import itemgetter
import grpc
from datetime import datetime
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

version = "tfserver" # "tfserver" or "tfserver_debug"

def get_point_by_points(line1,line2):
    a,b,c,d = line1
    e,f,g,h = line2
    ca = c-a
    db = d-b
    ge = g-e
    hf = h-f
    deno = db*ge-ca*hf
    fgeh = (f*g-e*h)/deno
    bcad = (b*c-a*d)/deno
    x = fgeh*ca-bcad*ge
    y = fgeh*db-bcad*hf
    return (int(x),int(y))

def get_ratio(bef, nex):
    hb, wb = bef[:2]
    hn, wn = nex[:2]
    return (hb/hn, wb/wn)

def edgeline_indexs(contours):
    linelens = []
    for i in range(len(contours)):
        u = contours[i][0] - contours[(i+1)%len(contours)][0]
        linelens.append(np.abs(np.linalg.norm(u)))
    return np.sort(np.argsort(linelens)[-4:])

def cv2tuple(points, hr=1.0, wr=1.0):
    return [(int(point[0][0]*wr), int(point[0][1]*hr)) for point in points]

def UNET_tfserver_debug(img):
    input_data = np.array(np.reshape(img/255, (1,256,256,1)), dtype=np.float32)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'unet_tf_256'  #モデル名
    request.model_spec.signature_name = 'serving_default'  #signature
    request.inputs['input'].CopyFrom(
        tf.make_tensor_proto(input_data, shape=[1,256,256,1]))   ##inputs
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    result = stub.Predict(request, 10)   #10秒のtimeout
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    img = np.array(result.outputs['output'].float_val).reshape(256,256)
    img[img > 0.3] = 1
    img[img <= 0.3] = 0
    img *= 255
    img = img.astype(np.uint8)
    return img

def UNET_tfserver(img):
    input_data = np.array(np.reshape(img/255, (1,256,256,1)), dtype=np.float32)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'unet_tf_256'  #モデル名
    request.model_spec.signature_name = 'serving_default'  #signature
    request.inputs['input'].CopyFrom(
        tf.make_tensor_proto(input_data, shape=[1,256,256,1]))   ##inputs
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Predict(request, 10)   #10秒のtimeout
    img = np.array(result.outputs['output'].float_val).reshape(256,256)
    img[img > 0.3] = 1
    img[img <= 0.3] = 0
    img *= 255
    img = img.astype(np.uint8)
    return img

def find_corner_unet(img):
    kernel = np.ones((3,3),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    index = np.argmax([cv2.contourArea(cnt) for cnt in cnts])
    cnt = cnts[index]
    convex = cv2.convexHull(cnt)
    epsilon = 0.005*cv2.arcLength(convex,True)
    approx = cv2.approxPolyDP(convex,epsilon,True)
    if len(approx) > 4:
        edgeindexs = edgeline_indexs(approx)
        choicedpoints = []
        for i in range(len(edgeindexs)):
            bef1= edgeindexs[(i-1)%len(edgeindexs)]
            bef2= (bef1+1)%len(approx)
            nex1= edgeindexs[i]
            nex2= (nex1+1)%len(approx)
            line1 = np.array([approx[bef1][0],approx[bef2][0]]).flatten()
            line2 = np.array([approx[nex1][0],approx[nex2][0]]).flatten()
            choicedpoints.append([list(get_point_by_points(line1, line2))])
        approx = np.array(choicedpoints)
    return approx

def find_corner(img):
    hr, wr = get_ratio(img.shape, (256,256,1))
    img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32)
    img = cv2.filter2D(img, -1, kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if version == "tfserver":
        fourpoint = cv2tuple(find_corner_unet(UNET_tfserver(gray)), hr=hr, wr=wr)
    elif version == "tfserver_debug":
        fourpoint = cv2tuple(find_corner_unet(UNET_tfserver_debug(gray)), hr=hr, wr=wr)
    else:
        print("set correct version (tfserver) or (tfserver_debug). now set as "+ str(version) + "." )
        return [(0,0),(0,0),(0,0),(0,0)]
    return fourpoint