import numpy as np

def compute_dsc(gt_mask, pred_mask):
    """
    :param gt_mask: Ground truth mask used as the reference
    :param pred_mask: Prediction mask

    :return: dsc=2*intersection/(number of non zero pixels)
    """
    numerator = 2 * np.sum(gt_mask*pred_mask)
    denominator = np.sum(gt_mask) + np.sum(pred_mask)
    if denominator == 0:
        # Both ground truth and prediction are empty
        return 0
    else:
        return numerator / denominator 