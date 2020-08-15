import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import numpy as np


def train_net(net, train_set):
    """
    训练模型，返回训练好的模型
    :param train_set:
    :param net:
    :return:
    """
    # 打印
    print("----------Start Training----------")

    # 训练集
    train_data, train_target = train_set

    if torch.cuda.is_available():
        net = net.cuda()
        train_data = train_data.cuda()
        train_target = train_target.cuda()

    # Dataloader
    train_set = Data.TensorDataset(train_data, train_target)
    train_loader = Data.DataLoader(dataset=train_set, batch_size=25, shuffle=True)

    # 优化器和损失函数
    opt = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.99), weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.cuda()

    # 整数据集训练次数
    epochs = 500
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(train_loader):
            train_output = net(data)
            loss = loss_func(train_output, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 打印精确度
            if (idx == 0) & ((epoch+1) % 8 == 0):

                test_data = train_data[200:400]
                test_target = train_target[200:400].data.numpy()
                test_output = net(test_data)
                pred_target = torch.max(test_output, 1)[1]
                pred_target = pred_target.data.numpy()
                accuracy = accuracy_score(pred_target, test_target)

                print("epoch: {:03}/{} | idx: {:03} loss: {:.4f} | accuracy: {:.4f}"
                      .format(epoch + 1, epochs, idx, loss.detach().numpy(), accuracy))

    return net


def ambiguity_detection(net, train_set, mark):
    """
    模糊识别，标记预测不出来的样本，是一个异常检测步骤
    :param mark:
    :param net:
    :param train_set:
    :return:
    """
    # 打印
    print("----------Start Ambiguity Detection----------")
    # 获得样本标签和预测
    train_data, train_target = train_set
    train_output = net(train_data)
    train_target = train_target.data.numpy()
    pred_target = torch.max(train_output, 1)[1].data.numpy()
    accuracy = accuracy_score(pred_target, train_target)
    print("accuracy: {}".format(accuracy))
    # 不一样就标记成ambiguous，即mark
    train_target = np.where(train_target == pred_target, train_target, mark)
    train_target = torch.from_numpy(train_target).type(torch.LongTensor)

    return train_data, train_target