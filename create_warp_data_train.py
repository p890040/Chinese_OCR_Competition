import cv2
import numpy as np
import os
import pandas as pd
import time
import json
import codecs
import shutil

root_path = r'E:\Research_code\myProject\Competition\Chinese_OCR\train'
path_data = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\img'
path_label = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\json'

# out_single = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\out_single'
# out_multiple = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\out_multiple'
# out_null = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\out_null'
out_single = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data0'
out_multiple = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data0'
out_null = r'E:\Research_code\myProject\Competition\Chinese_OCR\train\data0'
# shutil.rmtree(out_single, ignore_errors=True)
# os.makedirs(out_single, exist_ok=True)
# shutil.rmtree(out_multiple, ignore_errors=True)
# os.makedirs(out_multiple, exist_ok=True)
# shutil.rmtree(out_null, ignore_errors=True)
# os.makedirs(out_null, exist_ok=True)

out_null_count=0
img_names = []
label_names = []
charac= []
for i, anno in enumerate(os.listdir(path_label)):
    print(i)
    with codecs.open(os.path.join(path_label, anno), 'r', 'utf-8') as f:
      anno = json.load(f)
     
    img_name = anno['imagePath']
    img = cv2.imread(os.path.join(path_data, anno['imagePath']))
    show_img = img.copy()
    labels = anno['shapes']
    for j, label in enumerate(labels):

        pts = np.array(label['points'])
        pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
        pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
        xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])

        rect = cv2.minAreaRect(pts)
        box_p = np.int0(cv2.boxPoints(rect))
        angle= abs(rect[2])
        # print(box_p)
        
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        if(angle>45):
            width, height = height, width

        dst_pts = np.array([[0+rect[0][0]-width//2, height-1+rect[0][1]-height//2],
                            [0+rect[0][0]-width//2, 0+rect[0][1]-height//2],
                            [width-1+rect[0][0]-width//2, 0+rect[0][1]-height//2],
                            [width-1+rect[0][0]-width//2, height-1+rect[0][1]-height//2]], dtype="float32").astype(np.int32)


        over_p=0
        for _i in range(3):
            if(pts[0][0] > pts[_i+1][0]):
                over_p+=1
                
        if(over_p==2 and pts[0][1] == ymin):
            # print(over_p)
            # print(img_name[:-4]+'_'+str(i)+'.jpg')        
            _pts = pts.copy()
            _pts[0], _pts[1], _pts[2], _pts[3] = pts[3], pts[0], pts[1], pts[2]
            pts = _pts
        if(over_p==3):
            # print(over_p)
            # print(img_name[:-4]+'_'+str(i)+'.jpg')
            _pts = pts.copy()
            _pts[0], _pts[1], _pts[2], _pts[3] = pts[3], pts[0], pts[1], pts[2]
            pts = _pts

        pts2 = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]) - np.array([[xmin,ymin]])
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), pts2.astype(np.float32))
        dst = cv2.warpPerspective(img,M,(xmax-xmin,ymax-ymin))
        
        if(label['label']==''):
            assert len(label['label'])==0
            charac.append(label['label'])
            out_type = out_null
            out_null_count+=1
            cv2.polylines(show_img, [pts], isClosed=True, color = (255,0,0), thickness=1)
        elif(len(label['label'])==1):
            charac.append(label['label'])
            out_type = out_single
            cv2.polylines(show_img, [pts], isClosed=True, color = (0,255,0), thickness=1)
        else:
            for k, c in enumerate(label['label']):
                charac.append(c)
            out_type = out_multiple
            cv2.polylines(show_img, [pts], isClosed=True, color = (0,0,255), thickness=3)
            cv2.polylines(show_img, [box_p], isClosed=True, color = (0,255,255), thickness=3)
            cv2.polylines(show_img, [dst_pts], isClosed=True, color = (255,255,0), thickness=3)
            for z, _tmp in enumerate(dst_pts):
                cv2.putText(show_img, 'p'+str(z), (_tmp[0], _tmp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 0], 1, cv2.LINE_AA)

            
            

        # cv2.imwrite(os.path.join(out_type, img_name[:-4]+'_'+str(i)+'_'+str(j)+'.jpg'), img[ymin:ymax, xmin:xmax])
        # cv2.imwrite(os.path.join(out_type, img_name[:-4]+'_'+str(i)+'_'+str(j)+'.jpg'), dst)
        img_names.append(img_name[:-4]+'_'+str(i)+'_'+str(j)+'.jpg')
        label_names.append(label['label'])
    cv2.imwrite(os.path.join(r'E:\Research_code\myProject\Competition\Chinese_OCR\train\show2', img_name[:-4]+'_'+str(i)+'.jpg'),show_img)

        
# charac_uni = list(set(charac))
# predict_df = pd.DataFrame({'img_name': img_names, 
#                         'label_name': label_names,
#                         })
# predict_df.to_csv(os.path.join(root_path, "data.csv"), index=False, encoding='utf-8')


















    