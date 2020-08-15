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
    for i in range(8, 9):
        train_set = get_dataset("resources/dataset/set42/training_set@app42_time{}.csv".format(float(i/10)))

        target_base = get_target_base(train_set)
        train_set = target_str_to_idx(train_set, target_base)

        test_dataset = get_dataset("resources/dataset/set42/testing_set@app42_time{}.csv".format(float(i/10)))
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
    app = "21"
    for j in range(9, 10):
        time  = format(0.5+j/10, ".1f")
        net = torch.load("resources/models/app{}/model@{}app_time{}.pth".format(app, app, time),
                         map_location=torch.device('cpu'))
        target_base = np.load("resources/target_base/app{}/tb_model@{}app_time{}.npy".format(app, app, time))
        test_set = get_dataset("resources/dataset/app{}/testing_set@app{}_time{}.csv".format(app, app, time))
        test_set = target_str_to_idx(test_set, target_base)

        threshold_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        accuracy_list = []
        remain_rate_list = []
        for i in range(1, 200):
            precision, recall, f1, accuracy, remain_rate = test_net(net, test_set, len(target_base), 0.6+i*0.002)
            print("LeNet5 TEST | precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, accuracy: {:.4f}"
                  .format(precision, recall, f1, accuracy))

            threshold_list.append(0.6+i*0.002)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            remain_rate_list.append(remain_rate)

        draw(threshold_list, precision_list, recall_list, f1_list, accuracy_list, remain_rate_list, j+1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    plt.show()


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
    # main()
    print("precision: 0.5964, recall: 0.5157, f1: 0.4425, accuracy: 0.5521")
    print("precision: 0.5932, recall: 0.4310, f1: 0.4322, accuracy: 0.8422")