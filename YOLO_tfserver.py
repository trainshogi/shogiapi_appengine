import cv2
from PIL import Image
import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
from util import cv2pil, preprocess

debug = False

def yolo_tfserving(img):
    input_data = preprocess(img)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolov3'
    request.model_spec.signature_name = 'serving_default'  #signature
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(input_data, shape=[1,416,416,3]))   ##inputs
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Predict(request, 10)   #10秒のtimeout
    outputs = result.outputs
    boxes, scores, classes, nums = outputs["yolo_nms_0"].float_val, outputs[
        "yolo_nms_1_1"].float_val, outputs["yolo_nms_2_2"].float_val, outputs["yolo_nms_3_3"].int_val
    return boxes, scores, classes, nums

def yolo_tfserving_debug(img):
    input_data = preprocess(img)
    print(time.gmtime())
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolov3'
    request.model_spec.signature_name = 'serving_default'  #signature
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(input_data, shape=[1,416,416,3]))   ##inputs
    print(time.gmtime())
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print(time.gmtime())
    result = stub.Predict(request, 10)   #10秒のtimeout
    print(time.gmtime())
    outputs = result.outputs
    boxes, scores, classes, nums = outputs["yolo_nms_0"].float_val, outputs[
        "yolo_nms_1_1"].float_val, outputs["yolo_nms_2_2"].float_val, outputs["yolo_nms_3_3"].int_val
    return boxes, scores, classes, nums

def detect_img(img):
    result = []
    class_names = []
    if debug:
        out_boxes, out_scores, out_classes, num = yolo_tfserving_debug(img)
    else:
        out_boxes, out_scores, out_classes, num = yolo_tfserving(img)
    out_classes = np.array(out_classes,dtype="int")
    print(out_classes)
    for i in range(list(num)[0]):
        if debug:
            print(str(i)+": "+class_names[out_classes[i]])
            print(str(i)+": "+str(out_boxes[i]))
            print(str(i)+": "+str(out_scores[i]))
        if result == []:
            result.append([out_classes[i],1])
        elif result[-1][0] == out_classes[i]:
            result[-1][1] += 1
        else:
            result.append([out_classes[i],1])
    result.reverse()
    return result

def detect_mochigoma(gotemochi, sentemochi):
    gote = detect_img(cv2pil(gotemochi))
    sente = detect_img(cv2pil(sentemochi))    
    return gote, sente