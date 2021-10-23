# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:29:22 2021

@author: ying_yao
"""

import os
import numpy as np
import cv2
import pandas as pd

import codecs
import json 
import shutil 
import csv 
import tqdm 
import copy

def get_infos(data):
    category_list = data.label_name.unique().tolist()
    id_chara_map = {str(i):chara for i, chara in enumerate(category_list)}
    chara_id_map = {chara:str(i) for i, chara in enumerate(category_list)}
    return category_list, id_chara_map, chara_id_map

def merge_data(datas):
    img_names, label_names = [], []
    for data in datas:
        img_name = data.img_name.tolist()
        label_name = data.label_name.tolist()
        img_names.extend(img_name)
        label_names.extend(label_name)
    
    train_data = pd.DataFrame({'img_name': img_names, 
                               'label_id': label_names,
                               'label_name': label_names,
                               })
    return train_data

if __name__ == '__main__':
    out_root = r'E:\Research_code\myProject\Competition\Chinese_OCR\train'
    data_single= r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data_single.csv'
    data_addition_engdigit = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\addition_engdigit.csv'
    data_unknown = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data_unknown.csv'
    
    data_single = pd.read_csv(data_single)
    data_addition_engdigit = pd.read_csv(data_addition_engdigit)
    data_unknown = pd.read_csv(data_unknown)
    
    train_data = merge_data([data_single, data_addition_engdigit, data_unknown])
    
    
    category_list, id_chara_map, chara_id_map = get_infos(train_data)
    train_data['label_id'] = train_data['label_id'].map(chara_id_map)
    
    label_id_count = train_data['label_id'].value_counts()
    label_name_count = train_data['label_name'].value_counts()
    
    # with codecs.open(os.path.join(out_root, 'cls_label_map.json'), 'w', 'utf-8') as f:
    #     json.dump(id_chara_map, f)
    # train_data.to_csv(os.path.join(out_root, "train_cls.csv"), index=False, encoding='utf-8')
        

    
    # aaa,bbb=0,0
    # category_list+= ['a','odsfmn','123',':','9'] 
    # for c in category_list:
    #     try:
    #         c.encode(encoding='utf-8').decode('ascii')
    #         is_ch=False
    #     except:
    #         is_ch=True
    #     if(is_ch):
    #         aaa+=1
    #     else:
    #         bbb+=1
        