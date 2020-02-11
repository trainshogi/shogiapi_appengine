import cv2
import numpy as np
import grpc
from datetime import datetime
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
image_save = False

def CNN_tfserver(input_data):
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'cnn'  #モデル名
    request.model_spec.signature_name = 'serving_default'  #signature
    request.inputs['input'].CopyFrom(
        tf.make_tensor_proto(input_data, shape=[81,64,64,3]))   ##inputs
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    result = stub.Predict(request, 10)   #10秒のtimeout
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    return np.array(result.outputs['output'].float_val).reshape(81,-1)

def CNN_recognizer2(im,model=None):
    koma_class = [[0,1,3,5,7,9,11,13,15,17,19,21,23,27,31,33,35,37,39,41,43,45,47,49,51,53,55,59,63],# ichiji-kuro
                  [0,1,3,5,7,9,11,13,15,18,20,22,24,28,32,33,35,37,39,41,43,45,47,50,52,54,56,60,64],# ichiji-aka
                  [0,2,4,6,8,10,12,14,16,17,19,21,23,25,29,34,36,38,40,42,44,46,48,49,51,53,55,57,61],# niji-kuro
                  [0,2,4,6,8,10,12,14,16,18,20,22,24,26,30,34,36,38,40,42,44,46,48,50,52,54,56,58,62] # niji-aka
                 ]
    x = im
    x = np.asarray(x)
    x = x.astype('float32')
    x = x / 255.0
    result = []
    print("come to recognize")
    result_original = CNN_tfserver(x)
    result_first = np.argmax(result_original, axis=1)
    print(result_first)
    return result_first

def CNN_81pic(img_list):
    ban_result_2 = np.reshape(CNN_recognizer2(img_list),(9,9)).T
    return ban_result_2
