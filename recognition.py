import cv2, requests, base64, json, re
import numpy as np
from CNN_tfserver import CNN_81pic
from YOLO_tfserver import detect_mochigoma

def recognize(ban, goteimg, senteimg): 
    # url_ban   = "http://127.0.0.1:5001"
    # url_mochi = "http://127.0.0.1:5002"
    # #url_ban   = "http://pic2shogiapi-CNN.4qnthcfdte.ap-northeast-1.elasticbeanstalk.com"
    # #url_mochi = "http://pic2shogiapi-yolo.ap-northeast-1.elasticbeanstalk.com"
    # _,ban = cv2.imencode('.png',ban)
    # file_ban = {"upfile":base64.b64encode(ban)}
    # # goteimg = cv2.flip(goteimg,-1)
    # _,gote = cv2.imencode('.png',goteimg)
    # _,sente = cv2.imencode('.png',senteimg)
    # file_mochi = {"gote_data":base64.b64encode(gote),"sente_data":base64.b64encode(sente)}

    # res_ban = requests.post(url_ban+"/recognize",files=file_ban)
    # res_mochi = requests.post(url_mochi+"/recognize",files=file_mochi)

    # print(res_ban.text)
    # # {"result": "[[0, 38, 40, 0, 0, 42, 40, 0, 36], [0, 0, 0, 48, 0, 0, 0, 0, 0], [0, 34, 34, 44, 34, 0, 0, 44, 34], [0, 0, 0, 0, 0, 34, 51, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0], [2, 2, 0, 2, 2, 2, 2, 0, 2], [0, 12, 0, 0, 0, 8, 0, 14, 0], [4, 6, 0, 10, 16, 10, 0, 6, 0]]"}
    # print(res_mochi.text)
    # # {"gote": {"13": "1", "11": "1", "9": "1", "3": "1", "1": "1"}, "sente": {"7": "1", "5": "1", "3": "1"}}
    # jl_ban = json.loads(res_ban.text)
    # flat_ban = np.array([int(s) for s in re.sub(r'[ \[\]]+', "", jl_ban["result"]).split(',')])
    # resed_ban = flat_ban.reshape([9,9])

    resed_ban = CNN_81pic(ban)
    gotemochi, sentemochi = detect_mochigoma(goteimg, senteimg)
    jl_mochi = {"gote":gotemochi,"sente":sentemochi}
    
    # mochi = json.loads(res_mochi.text)
    # jl_mochi = {"gote":[],"sente":[]}
    # for i in list(mochi["gote"].items()):
    #     jl_mochi["gote"].append([int(i[0]),int(i[1])])
    # for i in list(mochi["sente"].items()):
    #     jl_mochi["sente"].append([int(i[0]),int(i[1])])

    return resed_ban, jl_mochi #json.loads(res_mochi.text)
