import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
cleverhans_path = osp.join(this_dir,'cleverhans')
add_path(cleverhans_path)


