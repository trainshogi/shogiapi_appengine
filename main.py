# -*- coding: utf-8 -*-
# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import json
import numpy as np
from make81pic import make81pic
from recognition import recognize
from result_to_json import result_to_json
# from list_to_kif import list_to_kif

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    # response.headers.add('Access-Control-Allow-Origin','*')
    response.headers.add('Access-Control-Allow-Headers','Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods','GET,PUT,POST,DELETE,OPTIONS')
    return response

# /post にアクセスしたときの処理
@app.route('/recognize', methods=['GET', 'POST'])
def post2():
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して
        file_data = request.files['upfile'].read()
        rotate = request.form.get('hidden_rotate')
        sengo = request.form.get('hidden_sengo')
        print(rotate)
        print(sengo)
        img_list, gotemochi, sentemochi = make81pic(file_data,rotate)
        print("enable to make pics.")
        ban_npy, mochi_tuple = recognize(img_list, gotemochi, sentemochi)
        print("enable to recognize.")
        # kif=list_to_kif(ban_npy,mochi_tuple,sengo)
        # print(kif)
        # kif_json = {'<br>'.join(kif.splitlines()):"result"}
        merged_json = json.dumps(result_to_json(ban_npy,mochi_tuple,sengo))
        return app.response_class(merged_json, content_type='application/json')
        
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return "ERROR!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True) #,debug=True)
    # app.run(host='0.0.0.0',port=4091,threaded=True) # どこからでもアクセス可能に
