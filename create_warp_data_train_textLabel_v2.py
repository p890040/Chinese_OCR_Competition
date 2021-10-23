import cv2
import numpy as np
import os
import pandas as pd
import time
import json
import codecs
import shutil

from shapely.geometry import Polygon
from shapely.geometry import Point
import shapely

root_path = r'E:\Research_code\myProject\Competition\Chinese_OCR\train'
path_data = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\img'
path_label = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\json'

# train_text_detection = r'E:\AI_Vision_Project\OCR_rproject\Feature Detection4 Tool2\Images'
train_text_detection = r'E:\AI_Vision_Project\OCR_rproject\Instance Segmentation4 Tool3\Images'

new_out = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\new_out'
os.makedirs(new_out, exist_ok=True)

img_names = []
label_names = []
charac= []
mul_sin_pair=[]
mul_sin_pair_mis=[]

def change_points_order(pts, ymin):
    over_p=0
    for _i in range(3):
        if(pts[0][0] > pts[_i+1][0]):
            over_p+=1
            
    if((over_p==2 and pts[0][1] == ymin) or over_p==3):        
        _pts = pts.copy()
        _pts[0], _pts[1], _pts[2], _pts[3] = pts[3], pts[0], pts[1], pts[2]
        pts = _pts
    return pts

def change_minrect_points_order(pts, dst_pts, shift):
    
    dst_pts_s = [dst_pts.copy(),
                 np.array([dst_pts[3], dst_pts[0], dst_pts[1], dst_pts[2]]),
                 np.array([dst_pts[2], dst_pts[3], dst_pts[0], dst_pts[1]]),
                 np.array([dst_pts[1], dst_pts[2], dst_pts[3], dst_pts[0]]),
                 ]
    shift_s = [shift.copy(),
               np.array([shift[3], shift[0], shift[1], shift[2]]),
               np.array([shift[3], shift[0], shift[1], shift[2]]),
               np.array([shift[3], shift[0], shift[1], shift[2]]),
               ]
    
    tmp = np.array([np.sum(np.linalg.norm(pts-dst_pts_s[0], axis=1)),
                    np.sum(np.linalg.norm(pts-dst_pts_s[1], axis=1)),
                    np.sum(np.linalg.norm(pts-dst_pts_s[2], axis=1)),
                    np.sum(np.linalg.norm(pts-dst_pts_s[3], axis=1)),
                    ])
    out = dst_pts_s[np.argmin(tmp)] - shift_s[np.argmin(tmp)]
    return out
    

segmentation_id=1
CATEGORIES=[{'id':1, 'name':'text', 'supercategory':'text'}]
coco_output = {"categories": CATEGORIES,"images": [],"annotations": []}
image_id=0

for i, anno in enumerate(os.listdir(path_label)):
    # print(i)
    with codecs.open(os.path.join(path_label, anno), 'r', 'utf-8') as f:
      anno = json.load(f)
    img_name = anno['imagePath']
    img = cv2.imread(os.path.join(path_data, anno['imagePath']))
    labels = anno['shapes']
    for j, label in enumerate(labels):
        pts = np.array(label['points'])
        pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
        pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
        xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])

        rect = cv2.minAreaRect(pts)
        box_p = np.int0(cv2.boxPoints(rect))
        angle = abs(rect[2])
        width = int(rect[1][0])
        height = int(rect[1][1])
        if(angle>45):
            width, height = height, width
        # dst_pts = np.array([[0, height-1],
        #                     [0, 0],
        #                     [width-1, 0],
        #                     [width-1, height-1]], dtype="float32")
        dst_pts = np.array([[0+rect[0][0]-width//2, height-1+rect[0][1]-height//2],
                            [0+rect[0][0]-width//2, 0+rect[0][1]-height//2],
                            [width-1+rect[0][0]-width//2, 0+rect[0][1]-height//2],
                            [width-1+rect[0][0]-width//2, height-1+rect[0][1]-height//2]], dtype="float32")
        shift = np.array([[rect[0][0]-width//2, rect[0][1]-height//2],
                          [rect[0][0]-width//2, rect[0][1]-height//2],
                          [rect[0][0]-width//2, rect[0][1]-height//2],
                          [rect[0][0]-width//2, rect[0][1]-height//2]], dtype="float32")
        
        dst_pts = change_minrect_points_order(pts, dst_pts, shift)
            
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst_pts.astype(np.float32))
        dst = cv2.warpPerspective(img, M, (width, height))
        


        # dst_show = dst.copy()
        
        # pts2 = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]) - np.array([[xmin,ymin]])
        # M = cv2.getPerspectiveTransform(pts.astype(np.float32), pts2.astype(np.float32))
        # M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst_pts.astype(np.float32))
        # dst = cv2.warpPerspective(img,M,(xmax-xmin,ymax-ymin))
        # dst = cv2.warpPerspective(img,M,(width,height))
        dst_show = dst.copy()

        
        if(label['group_id'] == 0 or label['group_id'] == 3): #0=>中文字串 3=>中英數字串
            singles=''
            text_pair=[]
            label['label'] = label['label'].replace('###','@')
            for mul_c in label['label']: 
                charc_pair=[]
                for _, lb in enumerate(labels):
                    lb['label'] = lb['label'].replace('###','@')
                    if(mul_c == lb['label'] and lb['group_id'] != 4):
                        pts_sin = np.array(lb['points'])
                        pts_sin[:,0] = np.clip(pts_sin[:,0], a_min=0, a_max=img.shape[1])
                        pts_sin[:,1] = np.clip(pts_sin[:,1], a_min=0, a_max=img.shape[0])
                        
                        poly_sin = Polygon(pts_sin)
                        poly_mul = Polygon(pts)
                        if(poly_sin.area > poly_mul.area):continue
                        intersect = poly_mul.intersection(poly_sin).area / poly_sin.area
                        
                        
                        if(intersect > 0.7):
                            new_single_pts = cv2.perspectiveTransform(pts_sin.astype(np.float32)[None,...], M).astype(np.int32)[0]
                            new_single_pts[:,0] = np.clip(new_single_pts[:,0], a_min=0, a_max=img.shape[1])
                            new_single_pts[:,1] = np.clip(new_single_pts[:,1], a_min=0, a_max=img.shape[0])
                            sin_xmin, sin_ymin, sin_xmax, sin_ymax = np.min(new_single_pts[:,0]), np.min(new_single_pts[:,1]), np.max(new_single_pts[:,0]), np.max(new_single_pts[:,1])
                            new_single_pts = np.array([[sin_xmin,sin_ymin],
                                                       [sin_xmax,sin_ymax]])
                            cv2.rectangle(dst_show, tuple(new_single_pts[0]), tuple(new_single_pts[1]), (0,255,0), thickness=2)
                            charc_pair.append([mul_c, 
                                               lb['label'],
                                               lb,
                                               np.linalg.norm(np.mean(pts,0) - np.mean(pts_sin,0)),
                                               new_single_pts])
                if(len(charc_pair)>0):
                    text_pair.append(charc_pair[0])
                # if(len(charc_pair)>1):
                #     charc_pair.sort(key = lambda x:x[3])
                #     cv2.rectangle(dst_show, tuple(charc_pair[0][4][0]), tuple(charc_pair[0][4][1]), (0,255,0), thickness=2)
                #     text_pair.append(charc_pair[0])
                #     labels.remove(charc_pair[0][2])
                #     singles+= charc_pair[0][1]
                # if(len(charc_pair)==1):
                #     cv2.rectangle(dst_show, tuple(charc_pair[0][4][0]), tuple(charc_pair[0][4][1]), (0,255,0), thickness=2)
                #     text_pair.append(charc_pair[0])
                #     singles+= charc_pair[0][1]
            
            new_img_name = img_name[:-4]+'_'+str(i)+'_'+str(j)+'.jpg'
            mul_sin_pair.append([new_img_name, label['label'], singles, charc_pair])
            cv2.imwrite(os.path.join(new_out, new_img_name), dst_show)
            cv2.imwrite(os.path.join(train_text_detection, new_img_name), dst)
            

            image_info = {
                "id": image_id,
                "file_name": new_img_name,
                "width": dst.shape[1],
                "height": dst.shape[0],
                }
            coco_output["images"].append(image_info)
            for k in range(len(text_pair)):
                box = text_pair[k][-1].reshape(-1).astype(np.float64)
                box[2:] = box[2:] - box[0:2]
                
                if(box[2]<1 or box[3]<1):
                    print(new_img_name)
                seg = [[box[0], 
                        box[1],
                        box[0]+box[2], 
                        box[1],
                        box[0]+box[2], 
                        box[1]+box[3],
                        box[0], 
                        box[1]+box[3],
                        ]]
                annotation_info = {
                    "id": segmentation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": list(box),
                    'segmentation':seg,
                    "angle":-1
                }
                coco_output["annotations"].append(annotation_info)
                segmentation_id+=1
            image_id+=1

with open(os.path.join(train_text_detection, 'trainval.json'), 'w') as f:
    json.dump(coco_output, f)














    