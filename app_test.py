import json
import numpy as np
import cv2
from make81pic import make81pic
from recognition import recognize
from result_to_json import result_to_json

file_data = cv2.imread("target/ban25.jpg")
rotate = 0
sengo = "faise"
img_list, gotemochi, sentemochi = make81pic(file_data,rotate)
print("enable to make pics.")
ban_npy, mochi_tuple = recognize(img_list, gotemochi, sentemochi)
print("enable to recognize.")
merged_json = json.dumps(result_to_json(ban_npy,mochi_tuple,sengo))
print(merged_json)