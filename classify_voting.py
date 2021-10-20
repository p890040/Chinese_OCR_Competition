import numpy as np
import cv2
import torch
import codecs
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import yaml
import torch.utils.data as tud
from collections import Counter
import json

def get_model(ckpt, model_name, label_map):
    model_cls = timm.create_model(model_name, pretrained=True)
    nclass = len(label_map)

    if(model_name == 'tf_efficientnetv2_m_in21k' or model_name == 'tf_efficientnetv2_s_in21k' or model_name == 'tf_efficientnetv2_l_in21k'):
        num_ftrs = 1280
        model_cls.classifier = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    elif (model_name == 'vit_base_patch32_224_in21k' or model_name == 'vit_base_r50_s16_224_in21k'):
        num_ftrs = 768
        model_cls.head = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    elif(model_name == 'resnest101e'):
        num_ftrs = 2048
        model_cls.fc = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    else:
        raise "Error"

    model_cls = model_cls.cuda()
    model_cls.load_state_dict(torch.load(ckpt))
    model_cls.eval()
    return model_cls


class Detect_Dataset(tud.Dataset):
    def __init__(self, detect_json, img_dir, transforms, resizes):
        super(Detect_Dataset, self).__init__()
        self.img_dir = img_dir
        print(f'[{type(self).__name__}] img_dir: ', self.img_dir)
        
        with open(detect_json) as f:
            self.detections = json.load(f)
        
        boxes, img_names, img_orders, has_detections, outputs=[], [], [], [], []
        for i, (k, v) in enumerate(self.detections.items()):
            has_detection=True
            if(len(v)==0):
                v = [[-1,-1,-1,-1]]
                has_detection=False
            for j in range(len(v)): 
                boxes.append(np.array(v[j]).astype(np.int32))
                img_names.append(k)
                img_orders.append(j)
                has_detections.append(has_detection)
                outputs.append('###')
        self.samples = pd.DataFrame({'boxes':boxes, 'img_names':img_names,'img_orders':img_orders, 'has_detections':has_detections, 'outputs':outputs})
        self.samples_pos = self.samples[self.samples['has_detections']]
        self.samples_neg = self.samples[~self.samples['has_detections']]
        self.idx_pos = self.samples_pos.index.tolist()
        self.idx_neg = self.samples_neg.index.tolist()

        self.boxes, self.img_names, self.img_orders = self.samples_pos['boxes'].values, self.samples_pos['img_names'].values, self.samples_pos['img_orders'].values

        self.resizes = resizes
        self.transforms = transforms
        
        assert len(self.transforms) == len(self.resizes)

        self.len_images = len(self.img_names)
        print('[{type(self).__name__}] len_images: ', self.len_images)
        
    def __len__(self):
        return self.len_images
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_names[idx]))
        box = self.boxes[idx]
        img = img[box[1]:box[3], box[0]:box[2]]
        imgs=[]
        for resize, transform in zip(self.resizes, self.transforms):
            img_single = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_single = cv2.resize(img_single, resize)
            img_single = transform(img_single)
            imgs.append(img_single)
        real_idx = self.idx_pos[idx]
        return imgs, real_idx

dontcare = []
for i in range(ord('0'),ord('0')+10):dontcare.append(chr(i))
for i in range(ord('A'),ord('A')+26):dontcare.append(chr(i))
for i in range(ord('a'),ord('a')+26):dontcare.append(chr(i))

if __name__ == "__main__":

    # path = '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/warp_crop3'
    path = '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/warp_crop_private'
    transform = transforms.Compose([
        # tvt.CenterCrop(224),p
        transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_imagenet = transforms.Compose([
        # tvt.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with codecs.open('/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/cls_label_map.json', 'r', 'utf-8') as f:
        label_map = json.load(f)

    configs =[
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_96_color_per_TW.yml',
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_96_color_per.yml',
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_96_color_TW.yml',    
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_96_color.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_224_color_per_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_224_color_per.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_224_color_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_224_color.yml',    
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_128_color_per_TW.yml',    
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_128_color_per.yml',    
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_128_color_TW.yml',    
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_128_color.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_300_color_per_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_300_color_per.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_300_color_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/S_300_color.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_384_color_per_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_384_color_per.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_384_color_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/M_384_color.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Res_256_color_per_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Res_256_color_per.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Res_256_color_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Res_256_color.yml',    
    # '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Vit_224_color_per_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Vit_224_color_per.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Vit_224_color_TW.yml',    
    '/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/configs/Vit_224_color.yml',    
    ]

    '''Prepare arguments'''
    models, transforms, label_maps, resizes=[],[],[],[]
    for i, config in enumerate(configs):
        with open(config) as f:
            hyp = yaml.safe_load(f) 
        ckpt = os.path.join('/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/run', os.path.basename(config), 'weight', 'model_cls_final.pth')

        models.append(get_model(ckpt, hyp['model_name'], label_map))
        label_maps.append(label_map)
        resizes.append((hyp['im_size'], hyp['im_size']))
        if(hyp.get('mean_std') == 'imagenet'):
            transforms.append(transform_imagenet)
        else:
            transforms.append(transform)

    '''Dataset loader'''
    test_Dataset = Detect_Dataset('/home/solomon/public/Pawn/slmdptc/test_box_batch1_private.json', path, transforms=transforms, resizes=resizes)
    test_loader = tud.DataLoader(test_Dataset, batch_size=120, shuffle=False, num_workers=12, pin_memory=True)
    print(f'loader size : {len(test_loader)}')

    '''All models inference'''
    outputs=[[] for _ in range(len(models))]
    s_time = time.time()
    for itv, (imgs, idx) in enumerate(test_loader):
        print(itv)
        assert len(models)==len(transforms)==len(label_maps)==len(resizes)
        num_infer=len(models)
        for i in range(num_infer):
            model, label_map = models[i], label_maps[i]
            model.eval()
            with torch.no_grad():
                img_batches = imgs[i].cuda()
                out = model(img_batches)
                y_score, y_pred = torch.max(F.softmax(out,1), 1)
                charas = y_pred.cpu().tolist()

                charas = [str(c) for c in charas]
                charas = pd.Series(charas).map(label_map).tolist()
                charas = ['###' if c in dontcare else c for c in charas]  
                # charas = ['' if c in dontcare else c for c in charas]  

            outputs[i].extend(charas)
    print(time.time()-s_time)

    '''Voting'''
    vote_outputs = pd.DataFrame(np.array(outputs))
    final_vote_outputs = []
    for i in range(len(vote_outputs.columns)):
        # get top n most frequent names
        n = 1
        chara = vote_outputs[i].value_counts()[:n].index.tolist()
        chara = chara[0]
        if(chara in dontcare):
            raise "Weird"
        final_vote_outputs.append(chara)

    '''Postprocess'''
    test_Dataset.samples['outputs'][test_Dataset.idx_pos] = final_vote_outputs
    final_vote_outputs = test_Dataset.samples[['img_names', 'outputs']]
    grouped_df = final_vote_outputs.groupby("img_names")
    grouped_lists = grouped_df["outputs"].apply(list)
    grouped_lists = grouped_lists.reset_index()
    img_names = grouped_lists['img_names'].tolist()
    final_characs = grouped_lists['outputs'].tolist()

    orders = [int(b.split('_')[2].replace('.jpg','')) for b in img_names]
    img_names = [x for _, x in sorted(zip(orders, img_names))]
    final_characs = [x for _, x in sorted(zip(orders, final_characs))]

    final_outs = []
    for i , str_list in enumerate(final_characs):
        final_out = ''.join(str_list)
        if(final_out==''):
            raise 'Weird'
        if(len(list(set(final_out)))==1 and  list(set(final_out))[0]== '#'):
            final_out = '###'
        final_outs.append(final_out)
    sample_submission = pd.read_csv('/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/sample_submission.csv', encoding='utf-8', header=None)
    sample_submission[9] = final_outs
    sample_submission.to_csv('/home/solomon/public/Pawn/Others/OCRTest/Chinese_OCR/public/vote_result_batch1_private.csv', encoding='utf-8', index=False, header=None, line_terminator='\n')
    
    
    
    



    