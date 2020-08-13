source venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd MAttNet/MAttNet/cv
python mattnet_server.py --model_id rcnn_cmr_with_st
