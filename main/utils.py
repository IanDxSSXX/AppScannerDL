import csv
import numpy as np
import os
import random
import matplotlib.pyplot as plt


def get_dataset(file_name):
    """
    读取csv文件 [str(name),float(time),[v1,v2,...v54]]并扩展与reshape
    :param file_name: 文件名
    :return: data, target_str
    """
    print("--------Start Reading File---------")
    target = []
    data = []
    with open(file_name, "r") as f:
        flow_list = csv.reader(f, quoting=csv.QUOTE_NONE, delimiter='|', skipinitialspace=True)
        for item in flow_list:
            # 如果数据一行不是 [包名， 时间， 54维向量] 的分布，就不要
            if len(item) == 3:
                target.append(item[0])
                flows = []
                flow_size = item[2][1:-1]
                flow_size = flow_size.split(", ")
                for i in flow_size:
                    flows.append(float(i))
                # 扩展成1*32*32个特征
                flows = np.repeat(flows, 19)[:-2]
                flows = np.reshape(flows, (1, 32, 32))
                data.append(flows)

    data = np.array(data)
    print("Data Shape: "+str(data.shape))
    target_str = np.array(target)

    # 打乱
    train = list(zip(data, target_str))
    random.shuffle(train)
    data, target_str = zip(*train)
    data = np.array(data)
    target_str = np.array(target_str)

    return data, target_str


def get_target_base(dataset):
    """
    用来得到标签的基准数组，如dataset=[好，坏，好，一般],输出target_base=[好，坏，一般]
    :param dataset:
    :return:
    """
    _, target_str = dataset

    # 消除重复项，得到一个base
    target_base = np.array(list(set(target_str)))
    save_target_base(target_base)

    return target_base


def target_str_to_idx(dataset, target_base):
    """
    用来把字符串标签转化成数组下标标签，如dataset=[好，坏，好，一般],target_base=[好，坏，一般]，输出就为[0,1,0,2]
    :param dataset:
    :param target_base:
    :return:
    """
    data, target_str = dataset
    target_idx = []

    # 将原本包名作为target转化成target_base的下标，变成整型
    for idx, target in enumerate(target_str):
        target_idx.append(np.where(target_base == target)[0][0])
    target_idx = np.array(target_idx)

    return data, target_idx


def save_target_base(target_base):
    """
    动态保存target——base
    :param target_base:
    :return:
    """
    i = 0.8
    while True:
        # 如果模型存在了，继续下去，否则直接存
        if os.access("./resources/target_base/tb_model@42app_time{}.npy".format(i), os.F_OK):
            i = float(format(i + 0.1, ".1f"))
            continue
        np.save("./resources/target_base/tb_model@42app_time{}.npy".format(i), target_base)
        break


def draw(threshold_list, precision_list, recall_list, f1_list, accuracy_list, remain_rate_list, i):
    """
    画图
    :param threshold_list:
    :param precision_list:
    :param recall_list:
    :param f1_list:
    :param accuracy_list:
    :param remain_rate_list:
    :return:
    """
    if i <= 5:
        plt.subplot(5, 2, 2 * (i - 1) + 1)
    else:
        plt.subplot(5, 2, 2 * (i - 5))
    plt.ylim(0, 100)
    plt.plot(threshold_list, [i*100 for i in precision_list], linewidth=1, markevery=20, marker='*',
             markersize=3, label="Precision")
    plt.plot(threshold_list, [i*100 for i in recall_list], linewidth=1, markevery=20, marker='h',
             markersize=3, label="Recall")
    plt.plot(threshold_list, [i*100 for i in f1_list], linewidth=1, markevery=20, marker='s',
             markersize=3, label="F1")
    plt.plot(threshold_list, [i*100 for i in accuracy_list], linewidth=1, markevery=20, marker='X',
             markersize=3, label="Accuracy")
    plt.plot(threshold_list, [i*100 for i in remain_rate_list], linewidth=1, markevery=20, linestyle='--', marker='o',
             markersize=3, label="Flow Remain Rate")
    if (i == 5) | (i == 10):
        plt.xlabel("PPT")
    if (i == 1) | (i == 6):
        plt.ylabel("Classifier Performance(%)")
    plt.title("Burst of time {}s".format(0.5+(i-1)/10))
    plt.grid()
