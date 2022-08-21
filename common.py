import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from adet.config import get_cfg


def setup_cfg(config_file, confidence_threshold=0.3):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', 'ctw1500_testr_R_50.pth', 'MODEL.TRANSFORMER.INFERENCE_TH_TEST', '0.3'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def vis_bases(bases):
    basis_colors = [[2, 200, 255], [107, 220, 255], [30, 200, 255], [60, 220, 255]]
    bases = bases[0].squeeze()
    bases = (bases / 8).tanh().cpu().numpy()
    num_bases = len(bases)
    fig, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
    for i, basis in enumerate(bases):
        basis = (basis + 1) / 2
        basis = basis / basis.max()
        basis_viz = np.zeros((basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
        basis_viz[:, :, 0] = basis_colors[i][0]
        basis_viz[:, :, 1] = basis_colors[i][1]
        basis_viz[:, :, 2] = np.uint8(basis * 255)
        basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
        axes[i // 2][i % 2].imshow(basis_viz)
    plt.show()

