import pandas as pd
import time, os, shutil
import cv2
import numpy as np


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

table = pd.read_csv('E:\Research_code\myProject\Competition\Chinese_OCR\public\Task2_Public_String_Coordinate.csv', header=None)
table = table.to_numpy()
path = r'E:\Research_code\myProject\Competition\Chinese_OCR\public\img_public'
out_path = r'E:\Research_code\myProject\Competition\Chinese_OCR\public\warp_crop_3'
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
for i in range(len(table)):
    img_name = table[i][0]+".jpg"
    img = cv2.imread(os.path.join(path, img_name))
    
    pts = np.array(table[i][1:]).astype(np.float32).reshape(-1,2)
    pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
    pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
    xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])

# =============================================================================
#     over_p=0
#     for _i in range(3):
#         if(pts[0][0] > pts[_i+1][0]):
#             over_p+=1
#             
#     if(over_p==2 and pts[0][1] == ymin):
#         # print(over_p)
#         # print(img_name[:-4]+'_'+str(i)+'.jpg')        
#         _pts = pts.copy()
#         _pts[0], _pts[1], _pts[2], _pts[3] = pts[3], pts[0], pts[1], pts[2]
#         pts = _pts
#     if(over_p==3):
#         # print(over_p)
#         # print(img_name[:-4]+'_'+str(i)+'.jpg')
#         _pts = pts.copy()
#         _pts[0], _pts[1], _pts[2], _pts[3] = pts[3], pts[0], pts[1], pts[2]
#         pts = _pts
# 
#     pts2 = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]) - np.array([[xmin,ymin]])
#     M = cv2.getPerspectiveTransform(pts.astype(np.float32), pts2.astype(np.float32))
#     dst = cv2.warpPerspective(img,M,(xmax-xmin,ymax-ymin))
#     cv2.imwrite(os.path.join(out_path, img_name[:-4]+'_'+str(i)+'.jpg'), dst)
# =============================================================================
    
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
    cv2.imwrite(os.path.join(out_path, img_name[:-4]+'_'+str(i)+'.jpg'), dst)
    

'''The author's function'''
# =============================================================================
#         # pt1 = [table[i][1], table[i][2]]
#         # pt2 = [table[i][3], table[i][4]]
#         # pt3 = [table[i][5], table[i][6]]
#         # pt4 = [table[i][7], table[i][8]]
#         # degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度
#         # partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
# =============================================================================

'''Minrect'''
# =============================================================================
#         # rect = cv2.minAreaRect(table[i][1:].astype(np.int32).reshape(4,2))
#         # box_p = cv2.boxPoints((rect[0], rect[1], rect[2])).astype(np.int32)
#         # pt1, pt2, pt3, pt4 = box_p
#         # # partImg = getSubImage(rect, img)
#         # partImg = dumpRotateImage(img, rect[2], pt1, pt2, pt3, pt4)
#         partImg = rotated_mask_to_bbox([table[i][1:].astype(np.int32)], img)
# =============================================================================
    
#         image = Image.fromarray(partImg).convert('L')
#         image = np.array(image)
#         cv2.imwrite(os.path.join(out_path, img_name[:-4]+'_'+str(i)+'.jpg'), image)
    