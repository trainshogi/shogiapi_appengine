# -*- coding: utf-8 -*-
def result_to_json(ban,mochi,sengo):
    gote = mochi["gote"]
    sente = mochi["sente"]
    kanji=["一","二","三","四","五","六","七","八","九"]
    koma=[" ・",
        " 歩"," 香"," 桂"," 銀"," 金"," 角"," 飛"," 玉",
        " と"," 杏"," 圭"," 全"," 馬"," 龍",
        "v歩","v香","v桂","v銀","v金","v角","v飛","v玉",
        "vと","v杏","v圭","v全","v馬","v龍"]
    koma=[" ・",
        " 杏"," 圭"," 全"," 馬"," 龍","v歩","v香","v桂","v銀","v金"," 歩",
        "v角","v飛","v玉","vと","v杏","v圭","v全","v馬","v龍"," 香",
        " 桂"," 銀"," 金"," 角"," 飛"," 玉"," と"
        ]
    koma=[" ・",
        " 歩"," 歩"," 香"," 香"," 桂"," 桂"," 銀"," 銀"," 金"," 金"," 角"," 角"," 飛"," 飛"," 玉"," 玉",
        " と"," と"," 杏"," 杏"," 圭"," 圭"," 全"," 全"," 馬"," 馬"," 馬"," 馬"," 龍"," 龍"," 龍"," 龍",
        "v歩","v歩","v香","v香","v桂","v桂","v銀","v銀","v金","v金","v角","v角","v飛","v飛","v玉","v玉",
        "vと","vと","v杏","v杏","v圭","v圭","v全","v全","v馬","v馬","v馬","v馬","v龍","v龍","v龍","v龍"]
    mochigoma=["歩","歩","香","香","桂","桂","銀","銀","金","金","角","角","飛","飛"]
    kazu=["","","二","三","四","五","六","七","八","九","十","十一","十二","十三","十四","十五","十六","十七","十八"]

    gote_json = {}
    for i in range(len(gote)):
        gote_json[mochigoma[gote[i][0]]] = kazu[gote[i][1]]
    
    ban_json = {}
    for i in range(9):
        for j in range(9):
            ban_json["\""+str(j)+str(i)+"\""] = koma[ban[i][j]]
    
    sente_json = {}
    for i in range(len(sente)):
        sente_json[mochigoma[sente[i][0]]] = kazu[sente[i][1]]

    sengo_json = ""
    if sengo != "true":
        sengo_json = "後手番"
    else:
        sengo_json = "先手番"
    
    result_json = {}
    result_json['ban_result']  = ban_json
    result_json['sente_mochi'] = sente_json
    result_json['gote_mochi']  = gote_json
    result_json['teban']       = sengo_json

    return result_json
