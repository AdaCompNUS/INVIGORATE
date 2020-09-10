import os.path as osp
import sys

this_dir = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.join(this_dir, '../')
sys.path.insert(0, ROOT_DIR)
