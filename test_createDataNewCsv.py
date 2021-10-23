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

import csv 

def PolyArea(points):
    x= points[:,0]
    y= points[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


if __name__ == '__main__':
    path_traindata= r'E:\Research_code\myProject\Competition\Chinese_OCR\train'
    path_cropdata= r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data0'
    
    path_outputDataCsv= r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data_new.csv'
    
    path_trainImg= os.path.join(path_traindata, "img")
    path_trainJson= os.path.join(path_traindata, "json")
    
    lst_json= os.listdir(path_trainJson)
    lst_cropImgName= os.listdir(path_cropdata)
    
    # row_1st= ["img_name", "label_name", "bbox_w", "bbox_h", "group_id"]
    # resultCsv=[row_1st]
    prev_jsonName= None 
    resultDict= {"img_name":[], 
                  "label_name":[], 
                  "bbox_w":[], 
                  "bbox_h":[], 
                  "group_id":[],
                  "polygon_area":[],
                  "bbox_area":[]}
    
    for idxImg, imgname in enumerate(lst_cropImgName):
        print(idxImg, imgname)
        if imgname[-4:] != ".jpg": continue 
        
    
        nameSplit= imgname[:-4].split("_")
        #print(nameSplit)
        if len(nameSplit) != 4 or nameSplit[0]!="img": continue 
        idx_shapes= int(nameSplit[-1])
        
        
        jsonName= f"img_{nameSplit[1]}.json"
        print("    -> json file: ", jsonName, idx_shapes)
        if jsonName != prev_jsonName:
            with codecs.open(os.path.join(path_trainJson, jsonName), 'r', 'utf-8') as f:
                data = json.load(f)
            print("     Load New json file")
            polyInfos= data["shapes"]
            prev_jsonName= jsonName
        '''
            group_id : 
                        0 -> Mandarin string (char numb>1)
                        1 -> Mandarin char (char numb ==1)
                        2 -> English string or Numbers
                        3 -> Mandrain + English + Number string 
                        4 -> Mandrin char + string
                        5 -> Others (not belong to Mandarin, English or Number)
                        255 ->　Don't care (too blur or serverely occluded)
            label : Madarin GT
            points : shape (4,2) 
                [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        '''
        points= np.array(polyInfos[idx_shapes]['points'])
        label= polyInfos[idx_shapes]['label']
        group_id= polyInfos[idx_shapes]['group_id']
        
        bbox_w, bbox_h= max(points[:,0])-min(points[:,0]),max(points[:,1])-min(points[:,1])
        polyArea= PolyArea(points)
        bboxArea= bbox_w*bbox_h
        
        '''
        img= cv2.imread(os.path.join(path_cropdata, imgname))
        imgH, imgW, _= img.shape
        if abs(imgH-bbox_h)>2 or abs(imgW - bbox_w)>2:
            
            print(f"  Cropped Image {imgname} is not consitant with json: ", abs(imgH-bbox_h), abs(imgW - bbox_w))
            break
        '''
        resultDict["img_name"].append(imgname)
        resultDict["label_name"] .append(label)
        resultDict["bbox_w"].append(bbox_w)
        resultDict["bbox_h"].append(bbox_h)
        resultDict["group_id"].append(group_id)
        resultDict["polygon_area"].append(polyArea)
        resultDict["bbox_area"].append(bbox_w*bbox_h)
        
        
        # row= [imgname, label, bbox_w, bbox_h, group_id]
        # resultCsv.append(row)
    
    df = pd.DataFrame(resultDict)
    df.to_csv(path_outputDataCsv)
    
    '''
    with open('data_new.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write 2d array table 
        writer.writerows(resultCsv)
    '''
    
    # df = pd.read_csv (path_outputDataCsv)
    # df['group_id'][df['group_id']==255]=10
    # hist= df['group_id'].hist(bins=10)
    pass