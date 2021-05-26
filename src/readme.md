# Requirements
INVIGORATE is tested on:
**Ubuntu 18.04**
**ROS melodic**
**Python 2.7**

# Clone the code 
```
git clone 
git submodule init 
git submodule update --recursive
```

# Environment setup
INVIGORATE relies on 4 different neural networks. Their environments have to be setup respectively. It is recommended that they each have their separate conda environment or docker.
## Detectron2 
Follow the official instructions to install Detectron2

## MAttNet
Follow the official instruction to install MAttNet

## VMRN
Follow the official instruction to install VMRN

## INGRESS
Download the official INGRESS docker.

# Download models
# Detectron2
download model
put it in <root_dir>/model

## VMRN
download all_in_one_1_25_1407_0.python
put it in <root_dir>/mrt_detector/vmrn_old/output/res101/vmrdcompv1

## MAttNet
download mattnet pretrained model.
put it in <root_dir>/output/refcoco_unc/
download maskrcnn model
put it in <root_dir>/pyutils/mask-faster-rcnn/output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/

## INGRESS
Included in the INGRESS docker

## INVIGORATE observation models
download density estimation pickle file
put it in model

## Download INVIGORATE Dataset
Download invigorate dataset from here.

# Install
This repo is a ROS workspace. Now build it.
```
catkin build
source devel/setup.bash
```

# Run
Assuming environments for different neural network modules are setup, launch them in separate terminals
```
conda activate detectron2
bash launch_detectron2.sh
```
```
conda activate mattnet
bash launch_mattnet.sh
```
```
conda activate vmrn
bash luanch_vmrn.sh
```
```
dcoker run -it adacompnus/ingress:v1.1 /bin/bash
ingress
```

Now we are ready to run INVIGORATE! There are 100 scenarios in the provided dataset, we have included experiment results for scenario 1-10. To replicate a particular scenario,
```
cd src/scripts
python dataset_demo --scene_num 1
```
The result will be saved inside src/logs directory. The result produced should be equivalent to the result in the dataset.
