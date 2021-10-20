import numpy as np
import cv2
import os
import sys
sys.path.insert(0, '/home/solomon/public/Pawn')
sys.path.insert(0, '/home/solomon/public/Pawn/slmdptc')
sys.path.insert(0, '/home/solomon/public/Pawn/slmdptc/detectron2')
print('system path :')
print(sys.path)

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS=None

import detectron2
#from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor, DefaultPredictor_Batch
#from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances, register_semantic
from detectron2.modeling.postprocessing import PYTHON_INFERENCE
import time
import json
import copy
from detectron2.utils.logger import setup_logger
import detectron2.data.detection_utils as utils
import ctypes
import torch

from skimage import measure
import skimage

from slm_ini import torch_clear_gpu_memory
from detectron2.utils.memory_gather import MemoryGather
# torch.backends.cudnn.benchmark = True

exec_environment = 0
Sol_version= ""
root_model = ""
model_type = "default"
#setup_logger()
def read_classname_file(arg_path):
    arg_path = os.path.join(os.path.dirname(os.path.abspath(arg_path)), 'class_name.txt')
    class_name, is_use_angle, color = [], [], []
    with open(arg_path) as f:
        text = [s.split() for s in f.readlines()]
        if(text[0][0] != 'BackGround'): 
            print('FORMAT ERROR!')
            raise 'FORMAT ERROR!'
        for name in text[1:]:
            class_name.append(name[0])
            is_use_angle.append(name[1])
            color.append(name[3])
    return class_name, is_use_angle, color

def read_maskTxt_3(arg_path):
    contours = []
    with open(arg_path) as f:
        
        for line in f.readlines():
            if '#start' in line:
                contour=[]
                continue
            if('#end' in line):
                contours.append(contour)
                contour=[]
                continue
            content= line.strip().split(',')
            contour.append(int(content[0]))
            contour.append(int(content[1]))
    return contours

def _postprocessing_for_del_file(path):
    for file in os.listdir(path):
        try:
            if('events.out' in file):
                os.remove(os.path.join(path, file))
        except:
            print("Error while deleting file ", os.path.join(path, file))

''' By Ying '''
def read_classname_keypoints_file(arg_path):
    class_name, is_use_angle, color = [], [], []
    with open(arg_path) as f:
        text = [s.split() for s in f.readlines()]
        for name in text:
            class_name.append(name[0])
            is_use_angle.append(name[1])
            color.append(name[3])
    return class_name, is_use_angle, color            

def read_keypoints_text(datasets_path, arg_keypionts_name):  
    '''
    Return: 
        dict_MaskKeypts:{ mask_filename: [num_keypts, (x,y,v)]}
    '''    
    img_names = ""
    mask_filename= ""
    dict_MaskKeypts= dict()
    with open(datasets_path) as f:
        text = f.readlines()
        for idx, line in enumerate(text):
            if line[0]== "#":
                img_names= text[idx+1].strip()                                      
                continue
            if len(line.split()) == 1:
                if line.strip()== img_names:
                    continue            
                if line[-5:].strip()== ".png":
                    mask_filename= line.strip()
                    num_keypt= int(text[idx+1].strip())
                    dict_MaskKeypts[mask_filename]= np.zeros((num_keypt,3))
                    for idx_keypt in range(num_keypt):
                        element= text[idx+ 2+ idx_keypt].split()
                        for tmp_i in range(3):
                            dict_MaskKeypts[mask_filename][idx_keypt,tmp_i]= int(float(element[1+ tmp_i]))
    return dict_MaskKeypts

def read_keypoints_text_coco(datasets_path, arg_keypionts_name):  
    '''
    Return: 
        dict_MaskKeypts:{ mask_filename: [num_keypts, (x,y,v)]}
    '''    
    img_names = ""
    mask_filename= ""
    dict_MaskKeypts= dict()
    with open(datasets_path) as f:
        text = f.readlines()
        for idx, line in enumerate(text):
            if line[0]== "#":
                img_names= text[idx+1].strip()                                      
                continue
            if len(line.split()) == 1:
                if line.strip()== img_names:
                    continue            
                if line[-5:].strip()== ".png":
                    mask_filename= line.strip()
                    num_keypt= int(text[idx+1].strip())
                    dict_MaskKeypts[mask_filename]= np.zeros((num_keypt*3))
                    # dict_MaskKeypts[mask_filename]= []
                    for idx_keypt in range(num_keypt):
                        element= text[idx+ 2+ idx_keypt].split()
                        for tmp_i in range(3):
                            # dict_MaskKeypts[mask_filename][arg_keypionts_name.index(element[0]),tmp_i]= int(float(element[1+ tmp_i]))
                            try:
                                dict_MaskKeypts[mask_filename][arg_keypionts_name.index(element[0])*3+tmp_i]= int(float(element[1+ tmp_i]))
                            except:
                                print()
                                print()
                    # dict_MaskKeypts[mask_filename]
                            # dict_MaskKeypts[mask_filename].extend(int(float(element[1+ tmp_i])))
    return dict_MaskKeypts

# from shapely.geometry import LineString
# from shapely.ops import unary_union
# import matplotlib.pyplot as plt
# def equal_sample_points(points, n=10):
#     line = LineString(points)
#     distances = np.linspace(0, line.length, n)
#     points = [line.interpolate(distance) for distance in distances]
#     multipoint = unary_union(points)
#     multipoint = np.array(multipoint)
#     return multipoint

class ObjectConfig():
    def __init__(self, path):
        self.NAME = None        
        self.vars= dict()
        self.lst_int= ["min_dimension", "max_dimension", "max_iter", "save_by_second", "is_use_data_aug", "max_detections", "batch_size", "rpn_post_nms_top_n", "relative_bg", "pr_on", "use_er", "use_er_limit", 'use_er_smallest_limit_pertask', 'use_er_largest_limit_pertask']
        self.lst_float= ["test_score_thresh", "gpu_limit", "base_lr", "weight_decay", "relative_bg_value", "angle_loss_weight", "roi_filter_ratio"]
        self.lst_str= ["dataset_folder", "annotation_file", "fine_tune_checkpoint", "docolorjitter", "dohistogram", "doresizelow", "doflip", "dorotate", "donoise", "doshift", "dozoom", "use_er_dir", 'ac_size', 'ac_ratios', 'iou_thr', 'iou_label', "pre_topk_test", "post_topk_test", 'pre_topk_train', 'post_topk_train', 'd_batch', 'positive_fraction', 'nms_thr', 'lr_decrease']
        self.loadConfigFromFile(path)
        self.root_path= os.path.dirname(os.path.abspath(path))
        self.model_path= self.root_path
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def loadConfigFromFile(self, path):     
        self.vars["docolorjitter"] = "False,0.5,0.5,0.5,0.2,0.5"
        self.vars["dohistogram"] = "False,clahe,0.0"
        self.vars["doresizelow"] = "False,0.4,1.0"
        self.vars["doflip"] = "False,both,0.7"
        self.vars["dorotate"] = "False,180,0.7"
        
        with open(path) as json_data:            
            configParam = json.load(json_data)
        for key in list(configParam.keys()):
            if key in self.lst_int:        
                self.vars[key]= int(configParam[key])
            elif key in self.lst_float:
                self.vars[key]= float(configParam[key])
            elif key in self.lst_str:
                self.vars[key]= configParam[key]            
        
        self.root_path= os.path.dirname(os.path.abspath(path))
        self.model_path= self.root_path
        self.class_name, self.is_use_angle, _ =  read_classname_file(self.vars['annotation_file'])
        self.vars['fine_tune_checkpoint'] = os.path.join(self.model_path, 'weights_predict.caffemodel')
        self.display()
        
    def changeConfig(self, key, value):
        if key in self.lst_int:        
            self.vars[key]= int(value)
        elif key in self.lst_float:
            self.vars[key]= float(value)
        elif key in self.lst_str:
            self.vars[key]= value
        else:
            print("Key [{0}] does not exit in Config".format(key))

class MaskRCNN_PYTORCH():
    
    def __init__(self, arg_mode, arg_config_path, arg_environ_path, gpu_device=''):
        torch.set_num_threads(1)
        print('========================')
        print('number of threads : ', torch.get_num_threads())
        print('========================')

        self.environ_path = arg_environ_path
        self.ConfigPath= arg_config_path 
        self.config = ObjectConfig(arg_config_path)
#        self.init_model()
        self.isNeedLoadModel = True
        self.datasets_register_name = "links_"+time.ctime()
        self.arg_mode = arg_mode
        self.training = False if('detecting' in arg_mode) else True
        
        self.mode = 'mask'
        if('FeatureDetect' in arg_mode):
            self.mode = 'feature'
        elif('keypoint' in arg_mode):
            self.mode = 'keypoint'
            
        self.for_detect_coco=False
        print(f'[Logging] mode : {self.mode}')
        print(f'[Logging] Sol_version : {Sol_version}')
        print(f'[Logging] root_model : {root_model}')
        print(f'[Logging] model_type : {model_type}')
        self.gpu_device = str(gpu_device)
        print(f'[Logging] gpu_device : {gpu_device}')
    
    # OPTION ER
    def memory_replay_save_sample(self, cfg, pdl_name='ProjectDeepLearning.dlproj'):
        try:
            from time import strftime, localtime
            # get cl infos
            with open(os.path.join(cfg['dataset_folder'], '..', pdl_name), 'r') as pdl:
                pdl_read = json.load(pdl)
                
            date_now = strftime("%Y%m%d_%H%M", localtime())
            if os.path.exists(os.path.join(cfg['use_er_dir'], date_now)):
                print('\n[INFO] Date is the same, which is not permitted, delay the saving for 60 seconds...\n')
                time.sleep(60)
                date_now = strftime("%Y%m%d_%H%M", localtime())
            print('\n[INFO] Task Date: current->{}\n'.format(date_now))
                
            memgather = MemoryGather(base_dir=cfg['use_er_dir'], tool_name=['Unknown', 'instance'][0])
            memgather.moveimg2onefolder = True
        
            # add the new folder task
            pdl_read['ContinualDataSetFolders'].append(os.path.join(cfg['use_er_dir'], date_now))
            
            if 'use_er_smallest_limit_pertask' in cfg and 'use_er_largest_limit_pertask' in cfg:
                print('[INFO ER-PERTASK] Defined smallest limit: {}, largest limit: {}'.format(cfg['use_er_smallest_limit_pertask'],
                                                                                                    cfg['use_er_largest_limit_pertask']))
                memgather.gather_memory_task(cfg['dataset_folder'], ntask=1, 
                                             limit_all_task=cfg['use_er_limit'],
                                             smallest_image_total = cfg['use_er_smallest_limit_pertask'],
                                             limit_image_total    = cfg['use_er_largest_limit_pertask'],
                                             use_date_format_folder=True, 
                                             date_format_name=date_now)
            else:
                memgather.gather_memory_task(cfg['dataset_folder'], ntask=1, limit_all_task=cfg['use_er_limit'], use_date_format_folder=True, date_format_name=date_now)
            with open(os.path.join(cfg['dataset_folder'], '..', pdl_name), 'w') as outfile:
                json.dump(pdl_read, outfile)
        except:
            return
            
    def initail_cfg(self):
        self.cfg = get_cfg()
        if(self.mode == 'mask' or self.mode=='keypoint'):
            if(model_type == 'default'):
                #cfg_path = os.path.join(self.environ_path, r"Lib\{0}\slmdptc\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_101_FPN_3x.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml".format(Sol_version))
            elif(model_type == 'light'):
                #cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_V_19_FPN_3x.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_V_19_FPN_3x.yaml".format(Sol_version))
            elif(model_type == 'pro'):
                #cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_V_99_FPN_3x.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_V_99_FPN_3x.yaml".format(Sol_version))
        elif(self.mode == 'feature'):
            if(model_type == 'default'):
                #cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml".format(Sol_version))
            elif(model_type == 'light'):
                #cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-Detection/faster_rcnn_V_19_FPN_3x.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-Detection/faster_rcnn_V_19_FPN_3x.yaml".format(Sol_version))
            elif(model_type == 'pro'):
                #cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-Detection/faster_rcnn_V_99_FPN_3x.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-Detection/faster_rcnn_V_99_FPN_3x.yaml".format(Sol_version))
        self.cfg.merge_from_file(cfg_path)

# =============================================================================
#         #Rotated RCNN #Pawn rotated temp
#         self.cfg = get_cfg()
#         from detectron2.point_rend import add_pointrend_config
#         add_pointrend_config(self.cfg)
#         cfg_path = r'C:/PYTHON3.6/Lib/ver3_2_0/slmdptc/detectron2/configs/Base-RRCNN-FPN.yaml'
#         self.cfg.merge_from_file(cfg_path)
#         # self.cfg.MODEL.MASK_ON = True
# =============================================================================
      
# =============================================================================
#         # RPN only
#         self.cfg = get_cfg()
#         cfg_path = r'C:/PYTHON3.6/Lib/ver3_2_0/slmdptc/detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml'
#         self.cfg.merge_from_file(cfg_path)
# =============================================================================

        # PointRend
        if(self.config.vars.get('pr_on') == 1):
            print('[Logging] : POINT REND ON')
            self.cfg = get_cfg()
            from detectron2.point_rend import add_pointrend_config
            add_pointrend_config(self.cfg)
            if(model_type == 'default'):
                # cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml".format(Sol_version))
            elif(model_type == 'light'):
                # cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_V_19_FPN_3x_coco.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_V_19_FPN_3x_coco.yaml".format(Sol_version))
            elif(model_type == 'pro'):
                # cfg_path = os.path.join(self.environ_path, r"Lib/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_V_99_FPN_3x_coco.yaml".format(Sol_version))
                cfg_path = os.path.join(self.environ_path, "Pawn/{0}/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_V_99_FPN_3x_coco.yaml".format(Sol_version))

            # cfg_path = r'C:/PYTHON3.6/Lib/ver3_2_0/slmdptc/detectron2/configs/COCO-InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml'
            self.cfg.merge_from_file(cfg_path)
            self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(self.config.class_name)

        if self.mode=='keypoint':
            self.cfg.MODEL.KEYPOINT_ON = True
            self.cfg.MODEL.ANGLE_ON = False
            keypointname_file= os.path.join(self.config.vars['dataset_folder'], "Annotation","class_name_keypoint.txt")
            self.keypointsname,_,_= read_classname_keypoints_file(keypointname_file)            
            self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(self.keypointsname)   
      

        '''ADD BY BRILIAN'''
        self.cfg.TRAINAUG.DOCOLORJITTER = self.config.vars.get('docolorjitter')  
        self.cfg.TRAINAUG.DOHISTOGRAM = self.config.vars.get('dohistogram')
        self.cfg.TRAINAUG.DORESIZELOW = self.config.vars.get('doresizelow')
        self.cfg.TRAINAUG.DOFLIP = self.config.vars.get('doflip')
        self.cfg.TRAINAUG.DOROTATE = self.config.vars.get('dorotate')  
        if 'donoise' in self.config.vars:
            if eval(self.config.vars.get('donoise').split(',')[0]):
                self.cfg.TRAINAUG.DONOISE = self.config.vars.get('donoise')     
        if 'doshift' in self.config.vars:
            if eval(self.config.vars.get('doshift').split(',')[0]):
                self.cfg.TRAINAUG.DOSHIFT = self.config.vars.get('doshift')     
        if 'dozoom' in self.config.vars:
            if eval(self.config.vars.get('dozoom').split(',')[0]):
                self.cfg.TRAINAUG.DOZOOM = self.config.vars.get('dozoom')     
        if eval(self.cfg.TRAINAUG.DOCOLORJITTER.split(',')[0]) or \
            eval(self.cfg.TRAINAUG.DOHISTOGRAM.split(',')[0]) or \
            eval(self.cfg.TRAINAUG.DORESIZELOW.split(',')[0]) or \
            eval(self.cfg.TRAINAUG.DOFLIP.split(',')[0]) or \
            eval(self.cfg.TRAINAUG.DOROTATE.split(',')[0]) or \
                self.cfg.TRAINAUG.DONOISE != '' or \
                self.cfg.TRAINAUG.DOSHIFT != '' or \
                self.cfg.TRAINAUG.DOZOOM != '':
            self.cfg.TRAINAUG.USE_SOLTRANSFORM = True
            if(not self.training):
                self.cfg.TRAINAUG.USE_SOLTRANSFORM = False
            if self.mode == 'mask' or self.mode=='feature': self.cfg.TRAINAUG.USE_KEYPOINTS = 'Notkeypoint, but USE_MASK'
            if self.mode=='keypoint': self.cfg.TRAINAUG.USE_KEYPOINTS = True
        ''''''
        
        self.cfg.OUTPUT_DIR = self.config.model_path
        self.cfg.DATASETS.TRAIN = (self.datasets_register_name,)
        self.cfg.DATASETS.TEST = (self.datasets_register_name, )  
        self.cfg.DATALOADER.NUM_WORKERS =0
    
        self.cfg.INPUT.MIN_SIZE_TRAIN = (self.config.vars['min_dimension'],)
        self.cfg.INPUT.MAX_SIZE_TRAIN = self.config.vars['max_dimension']
        self.cfg.INPUT.MIN_SIZE_TEST = self.config.vars['min_dimension']
        self.cfg.INPUT.MAX_SIZE_TEST = self.config.vars['max_dimension']
    
        self.cfg.SOLVER.MAX_ITER = (self.config.vars["max_iter"])  
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = self.config.vars.get('base_lr')
        self.cfg.SOLVER.WEIGHT_DECAY = self.config.vars.get('weight_decay')
        
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.class_name)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.vars['test_score_thresh']
        self.cfg.MODEL.ANGLE_ON = True if('True' in self.config.is_use_angle) else False
        self.cfg.MODEL.RPN.relative_bg = False if(self.config.vars['relative_bg'] == 0) else True
        self.cfg.MODEL.ROI_HEADS.relative_bg = False if(self.config.vars['relative_bg'] == 0) else True
        self.cfg.MODEL.RPN.relative_bg_value =  self.config.vars.get('relative_bg_value')
        self.cfg.MODEL.ROI_HEADS.relative_bg_value = self.config.vars.get('relative_bg_value')
        self.cfg.MODEL.WEIGHTS = os.path.join(self.config.model_path, 'model_final.pth') if(os.path.exists(os.path.join(self.config.model_path, 'model_final.pth'))) else root_model

        self.cfg.TEST.DETECTIONS_PER_IMAGE = self.config.vars.get('max_detections')
        # self.cfg.MODEL.ROI_ANGLE_HEAD.FC_DIM = 72*self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cfg.MODEL.ROI_ANGLE_HEAD.FC_DIM = 1024
        self.cfg.MODEL.ROI_ANGLE_HEAD.NUM_FC = 0
        
        self.cfg.MODEL.MASKPOINT_ON = False
        self.cfg.MODEL.ROI_MASKPOINT_HEAD.NUM_KEYPOINTS = 50
        self.cfg.MODEL.BMASK_ON = False
        if(self.cfg.MODEL.BMASK_ON):self.cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
        self.cfg.MODEL.MASKIOU_ON = False    
        self.cfg.MODEL.ITERDET_ON = False
    

        if(self.config.vars.get('pre_topk_test') is not None and self.config.vars.get('pre_topk_test') != ""):
            self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = int(self.config.vars.get('pre_topk_test'))
        if(self.config.vars.get('post_topk_test') is not None and self.config.vars.get('post_topk_test') != ""):
            self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = int(self.config.vars.get('post_topk_test'))
        if(self.config.vars.get('pre_topk_train') is not None and self.config.vars.get('pre_topk_train') != ""):
            self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = int(self.config.vars.get('pre_topk_train'))
        if(self.config.vars.get('post_topk_train') is not None and self.config.vars.get('post_topk_train') != ""):
            self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = int(self.config.vars.get('post_topk_train'))
        if(self.config.vars.get('ac_size') is not None and self.config.vars.get('ac_size') != ""):
            self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[int(o) for o in self.config.vars.get('ac_size').split(',')]]
        if(self.config.vars.get('ac_ratios') is not None and self.config.vars.get('ac_ratios') != ""):
            self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[float(o) for o in self.config.vars.get('ac_ratios').split(',')]]
        if(self.config.vars.get('iou_thr') is not None and self.config.vars.get('iou_thr') != ""):
            self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [float(o) for o in self.config.vars.get('iou_thr').split(',')]
        if(self.config.vars.get('iou_label') is not None and self.config.vars.get('iou_label') != ""):
            self.cfg.MODEL.ROI_HEADS.IOU_LABELS = [int(o) for o in self.config.vars.get('iou_label').split(',')]
        if(self.config.vars.get('d_batch') is not None and self.config.vars.get('d_batch') != ""):
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(self.config.vars.get('d_batch'))
        if(self.config.vars.get('positive_fraction') is not None and self.config.vars.get('positive_fraction') != ""):
            self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = float(self.config.vars.get('positive_fraction'))
        if(self.config.vars.get('nms_thr') is not None and self.config.vars.get('nms_thr') != ""):
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(self.config.vars.get('nms_thr'))
        if(self.config.vars.get('lr_decrease') is not None and self.config.vars.get('lr_decrease') != ""):
            self.cfg.SOLVER.STEPS = tuple([int(o) for o in self.config.vars.get('lr_decrease').split(',')])
        
        self.cfg.MODEL.DEVICE = 'cuda' if self.gpu_device=='' else 'cuda'+':'+ self.gpu_device
        
        # self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000

        # self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3, 0.7]
        # self.cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, -1, 1]
        # self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION  = 0.75
        # self.cfg.MODEL.BACKBONE.FREEZE_AT = 0
        # self.cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
        
        
        self.display_key_params()
        


    def display_key_params(self):
        print("[Architecture config] :")
        print(f'MASK : {self.cfg.MODEL.MASK_ON}')
        print(f'ANGLE : {self.cfg.MODEL.ANGLE_ON}')
        print(f'batch size : {self.cfg.SOLVER.IMS_PER_BATCH}')
        print(f'base_lr : {self.cfg.SOLVER.BASE_LR}')
        print(f'weight_decay : {self.cfg.SOLVER.WEIGHT_DECAY}')
        print(f'lr_decrease : {self.cfg.SOLVER.STEPS}')
        print(f'pre_topk_test : {self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST}')
        print(f'post_topk_test : {self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST}')
        print(f'pre_topk_train : {self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN}')
        print(f'post_topk_train : {self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN}')
        print(f'ac_size : {self.cfg.MODEL.ANCHOR_GENERATOR.SIZES}')
        print(f'ac_ratios : {self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}')
        print(f'iou_thr : {self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS}')
        print(f'iou_label : {self.cfg.MODEL.ROI_HEADS.IOU_LABELS}')
        print(f'd_batch : {self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}')
        print(f'positive_fraction : {self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION}')
        print(f'nms_thr : {self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}')    
        print(f'gpu_device : {self.cfg.MODEL.DEVICE}')
        pass
        
    # OPTION ER
    def check_if_er_exist(self):
        try:
            with open(os.path.join(self.config.vars['dataset_folder'], '..', 'ProjectDeepLearning.dlproj'), 'r') as pdl:
                pdl_read = json.load(pdl)
            # If there is a list of string in the key, then we will use training + ER
            if 'ContinualDataSetFolders' in pdl_read:
                if len(pdl_read['ContinualDataSetFolders']) > 0:
                    metadata = {'keypoint_names':self.keypointsname, 'keypoint_flip_map':[]} if self.mode=='keypoint' else {}
                    print("[ER INFO] CL images found, total data", len(pdl_read['ContinualDataSetFolders']))
                    self.datasets_register_name_ER = "links_"+time.ctime()
                    self.cfg.DATASETS.TRAIN_ER = tuple([self.datasets_register_name_ER])
                    register_coco_instances(self.datasets_register_name_ER, metadata, os.path.join(self.config.vars['use_er_dir'], 'annotations.json'), self.config.vars['use_er_dir'])
                else:
                    print('[ER INFO] No CL images available, proceed with normal training...')
        except Exception as e:
            print('[TRAIN ERROR] Message: ', e)
                
    def train(self, use_retrain=False):
        _postprocessing_for_del_file(self.config.model_path)
        # self.write_annotation()
        DatasetCatalog.clear()
        metadata={}
        if(self.mode=='keypoint'):metadata={'keypoint_names':self.keypointsname, 'keypoint_flip_map':[]}
        register_coco_instances(self.datasets_register_name, metadata, os.path.join(self.config.vars['dataset_folder'], 'trainval.json'), self.config.vars['dataset_folder'])
        self.initail_cfg()
        
        # OPTION ER
        self.check_if_er_exist()
        
        with open(os.path.join(self.config.vars['dataset_folder'],'logim.txt'), 'w') as f: pass # Pawn
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=True)
        
        self.trainer.train()
        
        # OPTION ER
        self.memory_replay_save_sample(self.config.vars)
        
        del self.trainer
        torch_clear_gpu_memory()
        
        _postprocessing_for_del_file(self.config.model_path)
        self.isNeedLoadModel = True

    def detect(self, arg_h, arg_w, arg_channel, arg_bytearray, use_gray=False):
        if(self.isNeedLoadModel):
            self.initail_cfg()
            self.cfg.MODEL.WEIGHTS = os.path.join(self.config.model_path, 'model_final.pth')
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0 if(self.for_detect_coco) else self.config.vars['test_score_thresh']
            self.predictor = DefaultPredictor(self.cfg)
            
            ''' custom Brilian'''
            if self.cfg.TRAINAUG.USE_SOLTRANSFORM:
                self.predictor.transform_gen = utils.build_soltransform(self.cfg, False)
            ''''''
            self.isNeedLoadModel=False
            self.fruits_nuts_metadata = MetadataCatalog.get(self.datasets_register_name).set(thing_classes=self.config.class_name)
        
        if(exec_environment==0 or self.for_detect_coco):
            if(use_gray):
                im = cv2.imread(arg_bytearray, 0)
            else:
                im = cv2.imread(arg_bytearray)
            PYTHON_INFERENCE[0] = True
        else:
            im= self.cvtCS2PyImg(arg_h, arg_w, arg_channel, arg_bytearray)
            im= cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            PYTHON_INFERENCE[0] = False
            
        
        outputs = self.predictor(im)
        
        
        predictions = outputs["instances"].to("cpu")
        boxes = predictions.pred_boxes.tensor.numpy()
        boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
        valid = np.where((boxes[:,2]-boxes[:,0]>0) & (boxes[:,3]-boxes[:,1]>0))[0] #Pawn. Select valid indice
        scores = predictions.scores.numpy()
        classes = predictions.pred_classes.numpy() + 1
        scores_angles = predictions.scores_angles.numpy() if predictions.has("scores_angles") else None
        # if(not(scores_angles is None)): print(np.argmax(scores_angles.reshape(-1, 9, 8), -1))
        
        boxes = boxes[valid,...]
        scores = scores[valid,...]
        classes = classes[valid,...]
        scores_angles = scores_angles[valid,...] if(not(scores_angles is None)) else scores_angles
        

        
        result = {"rois": boxes,
                  "class_ids": classes,
                  "scores": scores,
                  "angles_class" : scores_angles,
                  "angle_on" : self.cfg.MODEL.ANGLE_ON}
        
        if(self.mode == 'mask' or self.mode=='keypoint'):
            if(exec_environment==0):
                masks = np.array(predictions.pred_masks)
            else:
                if(PYTHON_INFERENCE[0]) : masks = np.array(predictions.pred_masks)
                else : masks = np.squeeze(np.array(predictions.pred_masks), axis=1)
            masks = masks[valid,...]
            result.update({"masks": masks})
            
        if(self.mode == 'keypoint'): 
            pred_keypoints = predictions.pred_keypoints.numpy()
            pred_keypoints = pred_keypoints[valid,...]
            result.update({"keypoints" : pred_keypoints})

        results=[result]
        return results[0]
    
    def write_annotation(self):
        IMAGE_DIR = self.config.vars['dataset_folder']
        SegmentImgs_DIR = os.path.join(IMAGE_DIR, "SegmentImgs")
        maskAnot_DIR = os.path.join(IMAGE_DIR, "Annotation/maskAnot.txt")
        datasets_DIR  = os.path.join(IMAGE_DIR, "Annotation/datasets.txt")
        class_name_DIR  = os.path.join(IMAGE_DIR, "Annotation/class_name.txt")
        class_name = read_classname_file(class_name_DIR)
        CATEGORIES=[{'id':i+1, 'name':_cls, 'supercategory':_cls} for i, _cls in enumerate(class_name[0])]
        coco_output = {"categories": CATEGORIES,"images": [],"annotations": []}
        
        result=dict()
        with open(datasets_DIR) as f:
            maskInfo=[]
            imgName="-1"
            text = f.readlines()
            for idx, line in enumerate(text):
                if line[0]== "#":
                    if idx !=0:
                        result[imgName]= copy.deepcopy(maskInfo)
                        maskInfo= list()
                    imgName= text[idx+1].strip()
                    continue
                if(len(line.split()) == 1):
                    continue
                l_s = line.split()
                maskInfo.append(l_s)
                result[imgName]= copy.deepcopy(maskInfo)
                
        image_id = 0
        segmentation_id = 1
        
        if(self.mode=='keypoint'):
            keypoints_file= os.path.join(self.config.vars['dataset_folder'], "Annotation","keypoints.txt") 
            keypointname_file= os.path.join(self.config.vars['dataset_folder'], "Annotation","class_name_keypoint.txt")
            self.keypointsname,_,_= read_classname_keypoints_file(keypointname_file)
            self.dict_MaskKeypt= read_keypoints_text_coco(keypoints_file, self.keypointsname)
        
        for key, value in result.items():
            image = cv2.imread(os.path.join(IMAGE_DIR, key))
            image_size = (image.shape[1], image.shape[0]) # W, H
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(key),
                "width": image_size[0],
                "height": image_size[1],
                }
            coco_output["images"].append(image_info)
            print(image_id)
            for l_s in value:
                try: cls_name = class_name[0][int(l_s[0])-1]
                except: cls_name = l_s[0]
                category_id = class_name[0].index(cls_name)+1
                bbox=[float(l_s[1]), float(l_s[2]), float(l_s[3])-float(l_s[1]), float(l_s[4])-float(l_s[2])]
                angle=float(l_s[5])

                annotation_info = {
                    "id": segmentation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "angle":int(angle)
                }

                if(self.mode=='mask' or self.mode=='keypoint' or self.mode=='feature'):
                    mask_txt = l_s[6].split('.')[0]+'.txt'
                    contours=read_maskTxt_3(os.path.join(SegmentImgs_DIR, mask_txt))
                    annotation_info.update({"segmentation": contours})
             
                import math
                
                def rotate(origin, point, angle):
                    """
                    Rotate a point counterclockwise by a given angle around a given origin.
                
                    The angle should be given in radians.
                    """
                    ox, oy = origin
                    px, py = point
                
                    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
                    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
                    return int(qx), int(qy)
                
                if(self.mode=='keypoint'):
                    annotation_info.update({"keypoints" : list(self.dict_MaskKeypt[l_s[6].split('.')[0]+'.png'])})

                coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1
            image_id = image_id + 1
        with open('{}/trainval.json'.format(IMAGE_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)  

    def cvtCS2PyImg(self, arg_h,arg_w, arg_channel, arg_bytearray):
        # Transfer image type from byte-array to numpy array
        byte_img = (ctypes.c_ubyte*arg_h*arg_w*arg_channel).from_address(int(arg_bytearray))
        img= np.ctypeslib.as_array(byte_img)
        img= np.reshape(img,(arg_h,arg_w,arg_channel))
        # [End]
        return img

    def del_self(self):
        print('del self.predictor')
        del self.predictor
        torch_clear_gpu_memory()
    
    def detect_output_coco(self, folder_name, img_names, out_json_path):
        class_names = read_classname_file(os.path.join(self.config.vars['dataset_folder'], 'Annotation/class_name.txt'))[0]

        self.detection_coco_results={}
        self.for_detect_coco=True
        img_names = list(img_names)
        for im_name in img_names:
            im_path = os.path.join(folder_name, im_name)
            print(f'[Logging] image : {im_name}')
            result = self.detect(None,None,None, im_path)
            wrap_detections(im_name, result, self.detection_coco_results, class_names)
        write_detection_annotation(self.detection_coco_results, class_names , out_json_path=out_json_path)
        self.for_detect_coco=False
    
        
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance).astype(np.int)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons



def wrap_detections(img_name, result, new_results, class_names):
    obj=[]
    boxes = result['rois']
    classes = result['class_ids']
    scores = result['scores']
    masks = result.get('masks')
    angles = result.get('angles')
    keypoints = result.get('keypoints')
    if angles is None:
        angles=np.ones_like(classes)*-1
    
    for i in range(boxes.shape[0]):
        if(masks is None): mask_poly = []
        else: mask_poly = binary_mask_to_polygon(masks[i])#[x1,y1,x2,y2,...]
        if(keypoints is None): keypoints=[]
        else: keypoint=keypoints[i]  
        obj.append([class_names[classes[i]-1],
                    boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],
#                    -1,
                    angles[i],
#                    mask_poly[0],
                    mask_poly,
                    scores[i],
                    keypoint
                    ])
    new_results[img_name] = obj

def write_detection_annotation(result, class_name , out_json_path=''):
    #image_size => W, H
    IMAGE_DIR = out_json_path
    CATEGORIES=[{'id':i+1, 'name':_cls, 'supercategory':_cls} for i, _cls in enumerate(class_name)]

    coco_output = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    image_id = 0
    segmentation_id = 1
    for key, value in result.items():
        image_size = cv2.imread(os.path.join(IMAGE_DIR, key)).shape[:2]
        image_size = (image_size[1], image_size[0]) # (W, H)
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(key),
            "width": image_size[0],
            "height": image_size[1],
            }
        coco_output["images"].append(image_info)
        print(image_id)
        for l_s in value:
            assert len(l_s)==8
            try:
                cls_name = class_name[int(l_s[0])-1]
            except:
                 cls_name = l_s[0]
            category_id = class_name.index(cls_name)+1
            # bbox=[float(l_s[1]), float(l_s[2]), float(l_s[3])-float(l_s[1]), float(l_s[4])-float(l_s[2])]
            bbox=[float(l_s[2]), float(l_s[1]), float(l_s[4])-float(l_s[2]), float(l_s[3])-float(l_s[1])]
            bbox=[int(b) for b in bbox]
            angle=l_s[5]
            contour=l_s[6]
            score = l_s[7]
            annotation_info = {
                "id": segmentation_id,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": 0,
                "bbox": bbox,
                "segmentation": contour,
                "score":str(score),
                "angle":int(angle)
            } 
            
            if(len(l_s[8])>0): #Has keypoint
                keypoint = list(l_s[8].reshape(-1).astype(np.float64))
                for i in range(len(keypoint)): 
                    if(i%3!=2): keypoint[i] = int(keypoint[i])
                annotation_info.update({"keypoints" : keypoint})
            
            
            coco_output["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1
        image_id = image_id + 1
    with open('{}/detection_annotation.json'.format(out_json_path), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)  



import codecs
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import yaml

def get_model(ckpt, model_name, label_map):
    model_cls = timm.create_model(model_name, pretrained=True)
    nclass = len(label_map)

    if(model_name == 'tf_efficientnetv2_m_in21k' or model_name == 'tf_efficientnetv2_s_in21k' or model_name == 'tf_efficientnetv2_l_in21k'):
        num_ftrs = 1280
        model_cls.classifier = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    elif (model_name == 'vit_base_patch32_224_in21k' or model_name == 'vit_base_r50_s16_224_in21k'):
        num_ftrs = 768
        model_cls.head = nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    elif(model_name == 'resnest101e'):
        num_ftrs = 2048
        model_cls.fc = nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    else:
        raise "Error"

    model_cls = model_cls.cuda()
    model_cls.load_state_dict(torch.load(ckpt))
    model_cls.eval()
    return model_cls

dontcare = []
for i in range(ord('0'),ord('0')+10):dontcare.append(chr(i))
for i in range(ord('A'),ord('A')+26):dontcare.append(chr(i))
for i in range(ord('a'),ord('a')+26):dontcare.append(chr(i))

def multi_inference(img, boxes, models=[], transforms=[], label_maps=[], resizes=[]):
    assert len(models)>0
    assert len(models)==len(transforms)==len(label_maps)==len(resizes)
    num_infer=len(models)
    if(boxes.shape[0]==0):
        return '###'
    if(img.shape[1] + 5>img.shape[0]): 
        boxes = boxes[boxes[:, 0].argsort()] # 水平長方形
    else: 
        boxes = boxes[boxes[:, 1].argsort()] #垂直長方形

    for i in range(num_infer):
        model, transform, label_map, resize = models[i], transforms[i], label_maps[i], resizes[i]
        model.eval()
        output=''
        for j in range(boxes.shape[0]):
            box = boxes[j]
            img_single =img[box[1]:box[3], box[0]:box[2]]

            img_single = cv2.resize(img_single, resize)
            img_single = Image.fromarray(img_single)
            with torch.no_grad():
                img_single = transform(img_single).unsqueeze(0).cuda()
                out = model(img_single)
                y_pred = F.softmax(out,1).argmax(1)
                chara = label_map[str(y_pred.item())]
                # print(chara)
                if(chara in dontcare):
                    pass
                else:
                    output+=chara
        if(output==''):
            output = '###'
        if(len(list(set(output)))==1 and  list(set(output))[0]== '#'):
            output = '###'
            
        return output

def multi_inference_voting(img, boxes, models=[], transforms=[], label_maps=[], resizes=[]):
    assert len(models)>0
    assert len(models)==len(transforms)==len(label_maps)==len(resizes)
    num_infer=len(models)
    if(boxes.shape[0]==0):
        return '###'
    if(img.shape[1] + 2>img.shape[0]): 
        boxes = boxes[boxes[:, 0].argsort()] # 水平長方形
    else: 
        boxes = boxes[boxes[:, 1].argsort()] #垂直長方形

    from collections import Counter
    vote_outputs=[]
    vote_outputs_scores=[]
    for i in range(num_infer):
        model, transform, label_map, resize = models[i], transforms[i], label_maps[i], resizes[i]
        model.eval()
        out_chara=[]
        out_chara_score=[]
        for j in range(boxes.shape[0]):
            box = boxes[j]
            img_single =img[box[1]:box[3], box[0]:box[2]]

            img_single = cv2.resize(img_single, resize)
            img_single = Image.fromarray(img_single)
            with torch.no_grad():
                img_single = transform(img_single).unsqueeze(0).cuda()
                out = model(img_single)
                # y_pred = F.softmax(out,1).argmax(1)
                y_score, y_pred = torch.max(F.softmax(out,1), 1)
                chara = label_map[str(y_pred.item())]
                # print(chara)

                if(chara in dontcare):
                    chara = '###'

                out_chara.append(chara)
                # out_chara_score.append([chara, y_score.item()])
        
        vote_outputs.append(out_chara)

    vote_outputs = pd.DataFrame(np.array(vote_outputs))
    final_out = ''
    for i in range(len(vote_outputs.columns)):
        # get top n most frequent names
        n = 1
        chara = vote_outputs[i].value_counts()[:n].index.tolist()
        chara = chara[0]
        if(chara in dontcare):
            pass
        else:
            final_out+=chara
    if(final_out==''):
        final_out = '###'
    if(len(list(set(final_out)))==1 and  list(set(final_out))[0]== '#'):
        final_out = '###'
    return final_out

        

        # vote_outputs_scores.append(out_chara_score)


            
        # return output


if __name__ == "__main__":

    Sol_version= ""
    root_model = r"D:\GitSource\Solvision\TaskProcess\bin\x64\Release\Models\InstanceSegment4\net1\weights_predict.caffemodel.pkl"    
    model_type = "default"
    # path = r'E:\AI_Vision_Project\OCR_rproject\Feature Detection4 Tool1\voc_config.json'
    path = '/home/solomon/public/Pawn/Others/OCRTest/CTWDataset/ctw_cropdata/voc_config.json'
    mrcnn_tool = MaskRCNN_PYTORCH('training_FeatureDetect', path, '/home/solomon/public')
    # path = '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/warp_crop3'
    path = '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/warp_crop_private'

    img_path_list = os.listdir(path)
    orders = [int(b.split('_')[2].replace('.jpg','')) for b in img_path_list]
    img_path_list = [x for _, x in sorted(zip(orders, img_path_list))]
    
    s_time = time.time()
    boxes_result={}
    for i, file in enumerate(img_path_list):
        if(i%20==0):
            print(f'({i}/{len(img_path_list)})')
        if(file[-4:]!='.png' and file[-4:]!='.jpg' and file[-4:]!='.bmp'): continue
        
        result = mrcnn_tool.detect(None,None,None, os.path.join(path, file), use_gray=False)
        boxes = result['rois'].astype(np.int32)
        boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
        img = cv2.imread(os.path.join(path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # outs.append(multi_inference_voting(img.copy(), boxes.copy(), models, transforms, label_maps, resizes))
        if(img.shape[1] + 2>img.shape[0]): 
            boxes = boxes[boxes[:, 0].argsort()] # 水平長方形
        else: 
            boxes = boxes[boxes[:, 1].argsort()] #垂直長方形

        boxes_result.update({file:boxes.tolist()})
    with open('test_box_batch24_private.json', 'w') as f:
        json.dump(boxes_result, f)
    print(f'total time:{time.time()-s_time:.3f}')
