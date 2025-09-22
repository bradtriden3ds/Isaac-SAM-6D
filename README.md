## UV Implementation 

Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Env
```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

PointNet2
```bash
cd SAM-6D
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
```

Download models
```bash
cd Instance_Segmentation_Model
python download_sam.py
python download_fastsam.py
python download_dinov2.py
cd ../

cd Pose_Estimation_Model
python download_sam6d-pem.py
```

## Demo
[Optional]
Download dataset via: https://huggingface.co/datasets/bop-benchmark/lm/tree/main

Env paths
```bash
cd SAM-6D/
export SAM_6D_FOLDER=/home/yizhou/Projects/SAM-6D/SAM-6D

# export CAD_PATH=$SAM_6D_FOLDER/Data/Example2/obj_000009.ply    
# export RGB_PATH=$SAM_6D_FOLDER/Data/Example2/rgb.png           
# export DEPTH_PATH=$SAM_6D_FOLDER/Data/Example2/depth.png       
# export CAMERA_PATH=$SAM_6D_FOLDER/Data/Example2/camera.json    
# export OUTPUT_DIR=$SAM_6D_FOLDER/Data/Example2/outputs   

export CAD_PATH=$SAM_6D_FOLDER/Data/Example3/cube.ply    
export RGB_PATH=$SAM_6D_FOLDER/Data/Example3/rgb.png           
export DEPTH_PATH=$SAM_6D_FOLDER/Data/Example3/depth.png       
export CAMERA_PATH=$SAM_6D_FOLDER/Data/Example3/camera.json    
export OUTPUT_DIR=$SAM_6D_FOLDER/Data/Example3/outputs  
```

Blender render
```bash
blenderproc run ./Render/render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 
```

Instance Segmentation
```bash
export SEGMENTOR_MODEL=sam
cd Instance_Segmentation_Model/
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH
```

Pose Estimation
```bash
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH --det_score_thresh=0.5
```

-----------------------------------------


# <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

####  <p align="center"> [Jiehong Lin](https://jiehonglin.github.io/), [Lihua Liu](https://github.com/foollh), [Dekun Lu](https://github.com/WuTanKun), [Kui Jia](http://kuijia.site/)</p>
#### <p align="center">CVPR 2024 </p>
#### <p align="center">[[Paper]](https://arxiv.org/abs/2311.15707) </p>

<p align="center">
  <img width="100%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/vis.gif"/>
</p>


## News
- [2024/03/07] We publish an updated version of our paper on [ArXiv](https://arxiv.org/abs/2311.15707).
- [2024/02/29] Our paper is accepted by CVPR2024!


## Update Log
- [2024/03/05] We update the demo to support [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), you can do this by specifying `SEGMENTOR_MODEL=fastsam` in demo.sh.
- [2024/03/03] We upload a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for running custom data.
- [2024/03/01] We update the released [model](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing) of PEM. For the new model, a larger batchsize of 32 is set, while that of the old is 12. 

## Overview
In this work, we employ Segment Anything Model as an advanced starting point for **zero-shot 6D object pose estimation** from RGB-D images, and propose a novel framework, named **SAM-6D**, which utilizes the following two dedicated sub-networks to realize the focused task:
- [x] [Instance Segmentation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)
- [x] [Pose Estimation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model)


<p align="center">
  <img width="50%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_sam_6d.png"/>
</p>


## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/JiehongLin/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
cd SAM-6D
sh prepare.sh
```
We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Evaluation on the custom data
```
# set the paths
export CAD_PATH=Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
cd SAM-6D
sh demo.sh
```



## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

