import numpy as np
from sklearn.neighbors import KDTree

def compress_model_with_error_keeping(pred, gt):
    '''
    The 3D predictions were compressed to less than 10M for ease of visualisation, 
    whilst retaining the correctly predicted and incorrectly predicted results in each category.
    '''
    total_approx_point_num = 400000 # can get 3d model around 10MB
    wrong_prediction_index = np.where(pred != gt)[0]
    correct_prediction_index = np.where(pred == gt)[0]

    pred_wrong = pred[wrong_prediction_index]
    pred_correct = pred[correct_prediction_index]

    uniq_wrong_label = np.unique(pred_wrong)
    uniq_correct_label = np.unique(pred_correct)

    max_cate_point_num = int(total_approx_point_num / (len(uniq_correct_label) + len(uniq_wrong_label)))

    wrong_select_index = []
    for uwl in uniq_wrong_label:
        uwl_index = np.where(pred_wrong == uwl)[0]
        number = len(uwl_index)
        if number > max_cate_point_num:
            sample_index = np.random.choice(np.arange(number), max_cate_point_num, replace=False)
            sample_index = uwl_index[sample_index]
        else:
            sample_index = uwl_index
        
        wrong_select_index.append(sample_index)
    
    wrong_select_index = np.concatenate(wrong_select_index)

    correct_select_index = []
    for ucl in uniq_correct_label:
        ucl_index = np.where(pred_correct == ucl)[0]
        number = len(ucl_index)
        if number > max_cate_point_num:
            sample_index = np.random.choice(np.arange(number), max_cate_point_num, replace=False)
            sample_index = ucl_index[sample_index]
        else:
            sample_index = ucl_index
        
        correct_select_index.append(sample_index)

    correct_select_index = np.concatenate(correct_select_index)

    total_select_index = np.concatenate([wrong_prediction_index[wrong_select_index], correct_prediction_index[correct_select_index]])

    return total_select_index

def back_proj_labels(sub_cloud, full_cloud, sub_pred, tree=None):
    if tree is None:
        tree = KDTree(sub_cloud)
    
    proj_index = np.squeeze(tree.query(full_cloud,  return_distance=False)).astype(np.int32)
    return sub_pred[proj_index]
