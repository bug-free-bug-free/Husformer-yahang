import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

def multiclass_acc(preds, truths):
    multi_acc = np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    return multi_acc

def get_preds(results):
    # 如果输出是分类概率 (batch, 3)
    if len(results.shape) == 2 and results.shape[1] == 3:
        preds = torch.argmax(results, dim=1).float()
        # 将分类索引 0, 1, 2 映射回原始标签 -1, 1, 2
        preds_mapped = preds.clone()
        preds_mapped[preds == 0] = -1
        preds_mapped[preds == 1] = 1
        preds_mapped[preds == 2] = 2
        return preds_mapped.cpu().detach().numpy()
    else:
        # 兼容 1 维回归的情况
        test_preds1 = results.view(-1).cpu().detach().numpy()
        for i, j in enumerate(test_preds1):
            if -1 < j < 0:
                test_preds1[i] = -1
            if 0 < j < 1:
                test_preds1[i] = 1
        test_preds1 = np.clip(test_preds1, a_min=-1., a_max=2.)
        return np.around(test_preds1)

def mae1(results, truths, exclude_zero=False):
    test_preds = get_preds(results)
    test_truth = truths.view(-1).cpu().detach().numpy()
    mae = np.mean(np.absolute(test_preds - test_truth))
    return mae

def eval_hus(results, truths, exclude_zero=False):
    test_preds = get_preds(results)
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))
    
    # 防止因预测出全零或其他单一常数导致相关系数计算报错
    try:
        corr = np.corrcoef(test_preds, test_truth)[0][1]
    except:
        corr = 0.0
        
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    _, _, f1, _ = precision_recall_fscore_support(
        test_preds[non_zeros], test_truth[non_zeros], average='weighted', zero_division=0)
    
    print("-" * 50)
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc: ", mult_a5)
    print('f1_score:', f1)
    print("-" * 50)