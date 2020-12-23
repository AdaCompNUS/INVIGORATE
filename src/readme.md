# Install
Download adacompnus/invigorate docker
Download adacompnus/ingress docker

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

# setup grasp policy
download density estimation pickle file
put it in grasp_planner

# Use it
follow launch_faster_rcnn_detector.sh to launch faster_rcnn_detector service in invigorate docker
follow launch_mattnet.sh to launch grounding service in invigorate docker
follow luanch_vmrn.sh to launch relationship detection service in invigorate docker
follow launch_ingress.sh to launch to launch ingress service in ingress docker