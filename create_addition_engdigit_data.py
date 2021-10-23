# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:25:21 2021

@author: lilin_chen
"""
import os, shutil
import numpy as np
import cv2
import pandas as pd
import random



root_path = r'E:\Research_code\myProject\Competition\Chinese_OCR\addition\eng_digit_data'
out_csv = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\addition_engdigit.csv'
out_path = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\output\eng_digit'
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
labes = []
for i in range(ord('0'),ord('0')+10):
    labes.append(chr(i))
for i in range(ord('A'),ord('A')+26):
    labes.append(chr(i))
for i in range(ord('a'),ord('a')+26):
    labes.append(chr(i))



step=0
final_data, final_label=[],[]
for i, folder in enumerate(os.listdir(root_path)):
    img_list= os.listdir(os.path.join(root_path, folder))
    random.shuffle(img_list)
    img_list = img_list[:5]
    for j in range(5):
        img = cv2.imread(os.path.join(root_path, folder, img_list[j]))
        cv2.imwrite(os.path.join(out_path, f'{step:04}'+'.jpg'), img)
        final_data.append(f'{step:04}'+'.jpg')
        final_label.append(labes[i])
        step+=1

predict_df = pd.DataFrame({'img_name': final_data, 
                        'label_name': final_label,
                        })
predict_df.to_csv(out_csv)
    
    

