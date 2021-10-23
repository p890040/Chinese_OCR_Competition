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

def PolyArea(points):
    x= points[:,0]
    y= points[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def copy_data(path, out_path, dataset):
    print(f"Copy frme {path} to {out_path}")
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)
    dataset = dataset['img_name'].tolist()
    # for i, file in enumerate(dataset):
    for file in tqdm.tqdm(dataset):
        shutil.copyfile(os.path.join(path, file), os.path.join(out_path, file))

if __name__ == '__main__':
    path_cropdata= r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data0'
    
    output_root= r'E:\Research_code\myProject\Competition\Chinese_OCR\train\output'
    output_multiple= os.path.join(output_root, 'multiple')
    output_multiple_Mand= os.path.join(output_multiple, 'Mandarin')
    output_multiple_Mand_Occlude= os.path.join(output_multiple, 'Mandarin_occl')
    output_multiple_MandEngNum= os.path.join(output_multiple, 'MandEngNum')
    output_multiple_EngNum= os.path.join(output_multiple, 'EngNum')
    output_single= os.path.join(output_root, 'single')
    output_unknown= os.path.join(output_root, 'unknown')
    output_empty= os.path.join(output_root, 'empty')
    
    path_DataCsv= r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data_new.csv'
    path_DataRoot= r'E:\Research_code\myProject\Competition\Chinese_OCR\train'
    
    path_data_Multple_mandarin= os.path.join(path_DataRoot, "data_Mandarin_NoOcclusion.csv")
    path_data_Multple_mandarin_occluded= os.path.join(path_DataRoot, "data_Mandarin_Occlusion.csv")
    path_data_Multple_MandEngNum= os.path.join(path_DataRoot, "data_ManEngNum.csv")
    path_data_Multple_EngNum= os.path.join(path_DataRoot, "data_EngNum.csv")
    path_data_Multple_unknown= os.path.join(path_DataRoot, "data_unknown.csv")
    path_data_Multple_empty= os.path.join(path_DataRoot, "data_empty.csv")
    
    
    path_data_1char= os.path.join(path_DataRoot, "data_single.csv")
    
    '''
    data_new.csv= {"img_name":[], 
                  "label_name":[], 
                  "bbox_w":[], 
                  "bbox_h":[], 
                  "group_id":[],
                  "polygon_area":[],
                  "bbox_area":[]}
    
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
    
    
    df = pd.read_csv (path_DataCsv)
    df['group_id'][df['group_id']==255]=10
    #hist= df['group_id'].hist(bins=10)
    # grp_5= df[(df['group_id']==5)]
    #grp_4= df[(df['group_id']==4) & (df['label_name'].str.len()==1)]
    
    
    #grp_1= df[(df['group_id']==1)]
    grp_0= df[(df['group_id']==0)]
    
    # grp_len1= df[df['label_name'].str.len()==1]
    # grp_AND= df[grp_4]
    #areaRatio= df['polygon_area']/df['bbox_area'] *100
    #hist_areaRatio= areaRatio.hist(bins=100)
    
    # Multiple
    grp_0_no_ocl= grp_0[ ~grp_0["label_name"].str.contains("###")]
    grp_0_ocl= grp_0[grp_0["label_name"].str.contains("###")]
    grp_3= df[(df['group_id']==3)]
    grp_2= df[(df['group_id']==2)]
    
    # Single
    grp_1_4= df[(df['group_id']==4) | (df['group_id']==1)]
    
    # Unkown
    grp_10 = df[(df['group_id']==10)]
    
    # Empty
    grp_5 = df[(df['group_id']==5)]
    
    
    grp_0_no_ocl.to_csv(path_data_Multple_mandarin)
    grp_0_ocl.to_csv(path_data_Multple_mandarin_occluded)
    grp_3.to_csv(path_data_Multple_MandEngNum)
    grp_2.to_csv(path_data_Multple_EngNum)
    grp_1_4.to_csv(path_data_1char)
    grp_10.to_csv(path_data_Multple_unknown)
    grp_5.to_csv(path_data_Multple_empty)
    
    print("="*10)
    copy_data(path_cropdata, output_multiple_Mand, grp_0_no_ocl)
    copy_data(path_cropdata, output_multiple_Mand_Occlude, grp_0_ocl)
    copy_data(path_cropdata, output_multiple_MandEngNum, grp_3)
    copy_data(path_cropdata, output_multiple_EngNum, grp_2)
    
    copy_data(path_cropdata, output_single, grp_1_4)
    
    copy_data(path_cropdata, output_unknown, grp_10)
    copy_data(path_cropdata, output_empty, grp_5)
    
    
    
    
    pass