# create cuda9.0 pytorch 1.4.0 tochvision 0.5.0 docker file

# cuda9.0

# install ros

# install miniconda and 
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
mkdir /root/.conda
bash Miniconda2-latest-Linux-x86_64.sh -b

# Edit .bashrc to resolve conflict with ROS

# Setup pytorch 0.4.0 environment for mattnet and vmrn_old
conda create -n pytorch_old python=2.7
pip install torch-0.4.0-cp27-cp27mu-linux_x86_64.whl

# setup faster_rcnn
download model
put it in <vmrn_root_dir>/output/vmrdcompv1/res101/

# setup vmrn_old
download all_in_one_1_25_1407_0.python
put it in <vmrn_old_root_dir>/output/res101/vmrdcompv1

# setup mattnet
## original mattnet with mask-rcnn
download mattnet pretrained model.
put it in <MAttnet-root-dir>/output/refcoco_unc/
download maskrcnn model
put it in <MAttnet-root-dir>/pyutils/mask-faster-rcnn/output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/

## mattnet v2
download mattnet pretrained model.
put it in <MAttnet-root-dir>/output/refcoco_small_unc/
