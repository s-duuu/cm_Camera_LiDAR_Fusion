# OpenPCDet_custom

  
    
    
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
    
    
## installation 
```
위에 들어가서 보면 spconv 설치하라는 거 아래에 setup.py develop 부분 해야 되는 거 그냥 하면 권한 문제 때문에 안됨

방법 1.
conda 환경에서 설치

방법 2.
python3 setup.py develop --user

이 방법 쓰면 numpy 랑 numba 버전 때문에 문제 생길 수 있음
numpy 버전은 1.20 써야 함!!!

pip3 install "numpy==1.20"

``` 
    
## 동작 logic (Nuvo)

### 0. conda activate 3d_lidar_torch

### 0-1. Dataset은 data/kitti/training에 kitti format으로 맞춰서 넣어주면 됨.

### 0-2. 넣고 난 이후에 다음 파일 실행해서 각종 파라미터 파일(pkl) 만들어주기(git 폴더 단에서)
```
python3 -m pcdet/datasets/kitti/kitti_dataset.py create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
### 1. cd 3d_lidar_OpenPCDet/tools

### 2. train
```
* .yaml 파일 넣을 때 data에 road_plane이 없다면 road_plane=False로 설정해 줘야 함!!!!

- gpu 1개
python3 train.py --cfg_file cfgs/kitti_models/pointpillar_newaugs.yaml

- gpu 2개
sh scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/pointpillar_newaugs.yaml
```
### 3. test
```
* nuvo에서 batch size는 4이하
python3 test.py --cfg_file cfgs/kitti_models/pointpillar_newaugs.yaml --batch_size ${BATCH_SIZE} --eval_all
```
### train, test 이후에 output 폴더가 tools 바깥 단에 생김 -> ckpt 폴더 내에 에폭 별 모델 파일 있음

### 4. demo
학습된 모델 파일 가져와서 테스트해보기
```
python3 demo_conti_draw.py --cfg_file cfgs/kitti_models/pointpillar_newaugs.yaml --ckpt trained_models/pointpillar_7728.pth --data_path ../data/kitti/test/000001.bin
```
```
(demo.py 는 수동, 기본 format)
python demo.py --cfg_file cfgs/kitti_models/pointpillar_newaugs.yaml \
    --ckpt {.pth} \
    --data_path ${POINT_CLOUD_DATA}
```



### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.
* All models are trained with 8 GTX 1080Ti GPUs and are available for download. 
* The training time is measured with 8 TITAN XP GPUs and PyTorch 1.5.

|                                             | training time | Car@R11 | Pedestrian@R11 | Cyclist@R11  | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) |~1.2 hours| 77.28 | 52.29 | 62.68 | [model-18M](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) | 
| [SECOND](tools/cfgs/kitti_models/second.yaml)       |  ~1.7 hours  | 78.62 | 52.98 | 67.15 | [model-20M](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view?usp=sharing) |
| [SECOND-IoU](tools/cfgs/kitti_models/second_iou.yaml)       | -  | 79.09 | 55.74 | 71.31 | [model-46M](https://drive.google.com/file/d/1AQkeNs4bxhvhDQ-5sEo_yvQUlfo73lsW/view?usp=sharing) |
| [PointRCNN](tools/cfgs/kitti_models/pointrcnn.yaml) | ~3 hours | 78.70 | 54.41 | 72.11 | [model-16M](https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing)| 
| [PointRCNN-IoU](tools/cfgs/kitti_models/pointrcnn_iou.yaml) | ~3 hours | 78.75 | 58.32 | 71.34 | [model-16M](https://drive.google.com/file/d/1V0vNZ3lAHpEEt0MlT80eL2f41K2tHm_D/view?usp=sharing)|
| [Part-A2-Free](tools/cfgs/kitti_models/PartA2_free.yaml)   | ~3.8 hours| 78.72 | 65.99 | 74.29 | [model-226M](https://drive.google.com/file/d/1lcUUxF8mJgZ_e-tZhP1XNQtTBuC-R0zr/view?usp=sharing) |
| [Part-A2-Anchor](tools/cfgs/kitti_models/PartA2.yaml)    | ~4.3 hours| 79.40 | 60.05 | 69.90 | [model-244M](https://drive.google.com/file/d/10GK1aCkLqxGNeX3lVu8cLZyE0G8002hY/view?usp=sharing) |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | ~5 hours| 83.61 | 57.90 | 70.47 | [model-50M](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing) |
| [Voxel R-CNN (Car)](tools/cfgs/kitti_models/voxel_rcnn_car.yaml) | ~2.2 hours| 84.54 | - | - | [model-28M](https://drive.google.com/file/d/19_jiAeGLz7V0wNjSJw4cKmMjdm5EW5By/view?usp=sharing) |
||
| [CaDDN (Mono)](tools/cfgs/kitti_models/CaDDN.yaml) |~15 hours| 21.38 | 13.02 | 9.76 | [model-774M](https://drive.google.com/file/d/1OQTO2PtXT8GGr35W9m2GZGuqgb6fyU1V/view?usp=sharing) |
