import os.path as osp
from pathlib import Path
import numpy as np

DATASET_DIR = "/run/media/alexander/T5 EVO/datasets"
FOLDER_TO_USE = Path(DATASET_DIR) / "ocean" / "Easy" / "P002"
POSES = FOLDER_TO_USE / "pose_left.txt"
GT_INTEGR =  FOLDER_TO_USE / "stamped_.txt"

poses = np.loadtxt(POSES, delimiter=" ")
gt_integrated =  np.loadtxt(GT_INTEGR, delimiter=" ")