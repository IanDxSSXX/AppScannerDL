from net import *
from test import *
from train import *
from utils import *
from fl import *


def main():
    a = 1
    if a == 0:
        retrain()
    elif a == 1:
        local_test()
    elif a == 2:
        fl_virtual_worker()



def retrain():
    """
    重新训练模型
    :return:
    """
    for i in range(10, 15):
        train_set = get_dataset("resources/dataset/training_set@app21_time{}.csv".format(float(i/10)))

        target_base = get_target_base(train_set)
        train_set = target_str_to_idx(train_set, target_base)

        test_dataset = get_dataset("resources/dataset/testing_set@app21_time{}.csv".format(float(i/10)))
        test_dataset = target_str_to_idx(test_dataset, target_base)

        net_ad = process(train_set, target_base)

        precision, recall, f1, accuracy, remain_rate = test_net(net_ad, test_dataset, len(target_base))
        print("FINAL TEST | flow remain rate: {:.4f} | precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, accuracy: {:.4f}"
              .format(remain_rate, precision, recall, f1, accuracy))


def local_test():
    """
    从本地拉下来模型来测试
    :return:
    """
    net = torch.load("resources/models/model@21app_time1.2.pth")
    target_base = np.load("resources/target_base/tb_model@21app_time1.2.npy")
    test_set = get_dataset("resources/dataset/testing_set@app21_time1.2.csv")
    test_set = target_str_to_idx(test_set, target_base)

    threshold_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    remain_rate_list = []
    for i in range(1, 200):
        precision, recall, f1, accuracy, remain_rate = test_net(net, test_set, len(target_base), 0.6+i*0.002)
        print("FINAL TEST | threshold: {:.3f} | flow remain rate: {:.4f} | "
              "precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, accuracy: {:.4f}"
              .format(0.6+i*0.002, remain_rate, precision, recall, f1, accuracy))

        threshold_list.append(0.6+i*0.002)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        remain_rate_list.append(remain_rate)

    draw(threshold_list, precision_list, recall_list, f1_list, accuracy_list, remain_rate_list)




def process(train_set, target_base):
    """
    综合训练
    :param train_set:
    :param target_base:
    :return:
    """
    # 打乱数据集
    train_data, train_target = train_set

    # numpy转tensor
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_target = torch.from_numpy(train_target).type(torch.LongTensor)

    # 分成初训练和模糊识别
    train_data_0 = train_data[:int(len(train_data) * 0.4)]
    train_target_0 = train_target[:int(len(train_target) * 0.4)]
    train_set_0 = (train_data_0, train_target_0)
    train_data_1 = train_data[int(len(train_data) * 0.4):]
    train_target_1 = train_target[int(len(train_target) * 0.4):]
    train_set_1 = (train_data_1, train_target_1)

    # 获得神经网络
    net = LeNet5(len(target_base))

    # 初次训练
    net = train_net(net, train_set_0)

    # 模糊识别返回新训练集
    train_set_2 = ambiguity_detection(net, train_set_1, len(target_base))

    # 重新训练
    net_ad = LeNet5(len(target_base) + 1)
    net_ad = train_net(net_ad, train_set_2)


    # 保存模型
    save_model(net_ad)

    return net_ad


if __name__ == "__main__":
    main()