# Chinese_OCR_Competition
Chinese OCR Competition in https://tbrain.trendmicro.com.tw/Competitions/Details/16

# Requirement
### Single Text Deteciton - Detectron2 (Object Detection)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```
### Character Classification - timm (pytorch-image-models)
```
pip install timm
```
### Other packages
```
pip install torch==1.9.0+cu111
pip install opencv-python==4.1.2.30
pip install pandas
pip install numpy
pip install shapely
```

### Script explanation
+ The scripts beginning with **create** are for producing our training or testing dataset.
+ text detection.py is **Single Text Deteciton** for generating rois.
+ train_v3.py, dataset.py, model.py and classify_voting.py is **Character Classification** for final recognition.
