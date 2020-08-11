import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_dataset
import numpy as np
import syft as sy
from syft.federated.floptimizer import Optims
hook = sy.TorchHook(torch)


def main():
    fl_virtual_worker()


def fl_virtual_worker():
    # 新的训练节点
    new_worker0 = sy.VirtualWorker(hook, id="new_worker0")
    new_worker1 = sy.VirtualWorker(hook, id="new_worker1")
    new_worker2 = sy.VirtualWorker(hook, id="new_worker2")


    # 从本地读数据集和模型
    net = torch.load("./resources/models/model@10apps.pth")
    target_base = np.load("resources/target_base/tb_model@10apps.npy")
    data, _ = get_dataset("resources/dataset/testing_set@time1.csv")
    # 当数据的准确度过了99%，就采样新的，作为半监督学习的数据集
    data, target = valid_dataset(net, data, len(target_base))
    data = torch.from_numpy(data).type(torch.FloatTensor)
    target = torch.from_numpy(target).type(torch.LongTensor)


    # 虚拟的工作者，所以这里发数据，返回的是指针，也就是无法获取真正的数据，做到联邦学习
    data_new_worker0 = data[:int(len(data)/3)].send(new_worker0)
    target_new_worker0 = target[:int(len(target)/3)].send(new_worker0)
    data_new_worker1 = data[int(len(data)/3):int(2*len(data)/3)].send(new_worker1)
    target_new_worker1 = target[int(len(data)/3):int(2*len(data)/3)].send(new_worker1)
    data_new_worker2 = data[int(2*len(data)/3):].send(new_worker2)
    target_new_worker2 = target[int(2*len(data)/3):].send(new_worker2)


    # 获取一个dataloader，包含所有工作者的信息
    fl_dataloader = [
        (data_new_worker0, target_new_worker0),
        (data_new_worker1, target_new_worker1),
        (data_new_worker2, target_new_worker2)
    ]

    # 将虚拟工作者放入list中
    new_workers = ["new_worker0", "new_worker1", "new_worker2"]


    # 为每个工作者创建一个自己的优化器，避免通过梯度而反推出来数据
    optims = Optims(new_workers, optim=torch.optim.Adam(params=net.parameters(), lr=0.0005))
    loss_func = nn.CrossEntropyLoss()


    epochs = 400
    for epoch in range(epochs):
        # 遍历每一个工作者的数据
        for data, target in fl_dataloader:
            # "我"把模型发给工作者
            net.send(data.location)
            # 获取对应的工作者的优化器
            opt = optims.get_optim(data.location.id)

            # 预测，出来的数据都是指针
            output = net(data)
            loss = loss_func(output, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 把更新好的模型发给"我"
            net.get()

            # 打印损失值
            print(loss.get())


def valid_dataset(net, data, mark, threshold=0.99):
    """
    半监督学习，过了阈值的就label
    :param net:
    :param data:
    :param mark:
    :param threshold:
    :return:
    """
    # data转torch
    data = torch.from_numpy(data).type(torch.FloatTensor)

    # 找到置信度最大的那一个target
    output = net(data)
    output = F.softmax(output, dim=1)
    pred = torch.max(output, 1)[1].data.numpy()
    # 找到最大的下标
    output = output.data.numpy()
    output_idx = np.amax(output, axis=1)
    # 小于阈值的走开
    pred = np.where(output_idx > threshold, pred, mark)
    # 若是mark，忽略
    length1 = len(data)
    idx = np.where(pred != mark)[0]

    # 获取最后过滤完的数据集
    data = data.data.numpy()[idx]
    target = pred[idx]
    length2 = len(data)
    remain_rate = length2 / length1
    print("过滤后的比例: {}".format(remain_rate))

    return data, target


if __name__ == '__main__':
    main()