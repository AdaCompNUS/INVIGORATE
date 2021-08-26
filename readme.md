# Requirements
INVIGORATE is tested on:
**Ubuntu 18.04**
**ROS melodic**
**Python 2.7**

# Clone the code
```
git clone https://github.com/AdaCompNUS/INVIGORATE.git
cd INVIGORATE
git submodule init
git submodule update --recursive
```

# Environment setup
INVIGORATE relies on 4 different neural networks. Their environments have to be setup respectively. It is recommended that they each have their separate conda environment or docker. Since MAttNet and VMRN requires pytorch 0.4 and CUDA 8.0, it is recommended to use our provided docker to run them.

## Detectron2
We use a custom version of detectron2. To install dependencies, do
```
conda create -n detectron2 python=3.8 # detectron uses python3
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
cd src/faster_rcnn_detector/detectron2
pip install -e .
pip install rospkg catkin_pkg opencv-python # to make ROS work with python3
```

## MAttNet and VMRN
We provide a docker to run MAttNet and VMRN. It contains a conda environment called torch_old that contains cuda 9.0 and pytorch 0.4.

1. Install [docker](https://docs.docker.com/engine/install/ubuntu/)
2. Follow the [official instruction](https://github.com/NVIDIA/nvidia-docker) to install nvidia-docker. This allows docker to use your GPU.
3. Download and run our provided [docker](https://hub.docker.com/repository/docker/adacompnus/vmrd)
```
cd .. # move to the parent folder of INVIGORATE
cp -r INVIGORATE INVIGORATE_docker # Make a copy of INVIGORATE. This is because we need to build the ROS workspace inside docker separately.
docker pull adacompnus/vmrd
docker run --gpus all -v "$(pwd)"/INVIGORATE_docker:/home/INVIGORATE_docker --network host -it adacompnus/vmrd /bin/bash # mount INVIGORATE_docker folder into docker and give GPU and network access
```
4. Now you are inside the docker, use the provided **torch_old** environment.
```
conda activate torch_old # the docker has a conda environment with cuda 9.0 and pytorch 0.4
cd /home/INVIGORATE_docker # go to mounted folder
catkin build
source devel/setup.bash
```

### MAttNet
Proceed to follow the [official instruction](https://github.com/lichengunc/MAttNet) to setup an environment for MAttNet. Some of the most important steps are shown below.
```
conda activate torch_old # if you have not done this.
cd src/mattnet_server/MAttNet
```
1. Install cocoapi
```
cd pyutils/mask-faster-rcnn
git clone https://github.com/cocodataset/cocoapi data/coco
cd data/coco/PythonAPI
make
```
2. compile models
```
cd pyutils/mask-faster-rcnn/lib
make
```
3. compile refer submodule
```
cd pytuils/refer
make
```

### VMRN
We are using a legacy version of [VMRN](https://github.com/ZhangHanbo/Visual-Manipulation-Relationship-Network-Pytorch/tree/pytorch0.4.0_oldversion).
Just go to vmrn_old submodule and run make.sh
```
conda activate torch_old # if you have not done this.
cd src/mrt_detector/vmrn_old
bash make.sh
```

## INGRESS (Required only if you want to generate captions for asking questions)
1. Install [docker](https://docs.docker.com/engine/install/ubuntu/)
2. Follow the [official instruction](https://github.com/NVIDIA/nvidia-docker) to install nvidia-docker. This allows docker to use your GPU.
3. Download the [official INGRESS docker](https://hub.docker.com/r/adacompnus/ingress) and run it
```
docker run --gpus all --network host -it adacompnus/ingress /bin/bash
```
4. Inside docker, install lua torch and cuda libraries:
```bash
$ luarocks install cutorch
$ luarocks install cunn
$ luarocks install cudnn
```
5. Make sure you can run [Demo](https://github.com/AdaCompNUS/ingress-proj) for INGRESS.

## INVIGORATE
Create a conda environment for INVIGORATE
```
conda create -n invigorate python=2.7
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

# Download models
All models are stored [here](https://drive.google.com/drive/folders/1jLva2HR6QLxKdaXZBxK4RKI_dNQBDZtK?usp=sharing)
# Detectron2
Download model_final_cascade.pth
put it in <root_dir>/src/model

## VMRN
Download all_in_one_1_25_1407_0.pth
put it in <root_dir>/src/mrt_detector/vmrn_old/output/res101/vmrdcompv1.
*(The root directory should be INVIGORATE_docker if you follow the procedure above.)*

## MAttNet
Download mattnet pretrained model on refCOCO from its official repository. *(The root directory should be INVIGORATE_docker if you follow the procedure above.)*

## INGRESS
Included in the INGRESS docker

## INVIGORATE observation models
Download density estimation pickle files. (ground_density_estimation_mattnet.pkl and relation_density_estimation.pkl)
put it in <root_dir>/src/model

## Download INVIGORATE Dataset
Download invigorate dataset from [here](https://drive.google.com/file/d/1FUoLSZupPi1J3BNRY2VTYC1bKWe50ZRF/view?usp=sharing).
Extract it and put it under the <root_dir>/src

# Install
This repo is a ROS workspace. Now build it.
```
catkin build
source devel/setup.bash
```

# Run
Assuming environments for different neural network modules are setup, launch them in separate terminals
1. Start roscore
```
roscore
```
2. Start detectron service
```
conda activate <detectron2_env> # use your environment name
bash launch_detectron2.sh
```
3. Start mattnet service inside docker
```
conda activate torch_old
bash launch_mattnet.sh
```
4. Start vmrn service inside docker
```
conda activate torch_old
bash luanch_vmrn.sh
```
5. Start INGRESS service inside docker (If you want to generate captions)
```
docker run -it adacompnus/ingress:v1.2 /bin/bash
ingress
```

Now we are ready to run INVIGORATE! There are 100 scenarios in the provided dataset, we have included experiment results for scenario 1-10. To replicate a particular scenario,
```
conda activate invigorate
cd src/scripts
python dataset_demo --scene_num 1
```
If you want to generate captions:
```
python dataset_demo --scene_num 1 --captions
```
The result will be saved inside src/logs directory. The result produced should be equivalent to the result in the dataset.
