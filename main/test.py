import torch
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def test_net(net, test_set, mark, threshold=0):
    """
    用已知模型测试数据
    :param net: 已知模型
    :param mark:
    :param threshold:
    :param test_set: 测试集
    :return: precision, recall, f1, accuracy, flow remain rate
    """
    # 测试集
    test_data, test_target = test_set
    # data转torch，target不用转
    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)

    # 找到置信度最大的那一个target
    test_output = net(test_data)
    test_output = F.softmax(test_output, dim=1)


    pred_target = torch.max(test_output, 1)[1].data.numpy()
    # 找到最大的下标
    test_output = test_output.data.numpy()
    test = np.amax(test_output, axis=1)
    # 小于阈值的走开
    pred_target = np.where(test > threshold, pred_target, mark)

    # 若是mark，忽略
    length1 = len(pred_target)
    idx = np.where(pred_target != mark)[0]
    test_target = test_target[idx]
    pred_target = pred_target[idx]
    length2 = len(pred_target)
    remain_rate = length2/length1

    # 精确度： tp/(tp+fp)
    precision = precision_score(test_target, pred_target, average="macro", zero_division=1)
    # 召回率：tn/(tn+fn)
    recall = recall_score(test_target, pred_target, average="macro", zero_division=1)
    # ps和rs的调和平均数
    f1 = f1_score(test_target, pred_target, average="macro", zero_division=1)
    # 正确分类的比例
    accuracy = accuracy_score(test_target, pred_target)

    return precision, recall, f1, accuracy, remain_rate