import torch
import torch.nn as nn
from card_number_identification.googlenet import GoogLeNet
from card_number_identification.data_loader import Dataset
import multiprocessing
import time
from torch.autograd import Variable
import os
from os.path import join, dirname
from copy import deepcopy as dp

# import hiddenlayer as hl

exp_name = 'exp2'

## 训练参数设置
batch_size = 10  # 批大小
num_epochs = 30  # 训练次数
lr_ = 0.1  # 学习率
weight_decay = 1e-4
steppoints = [8, 16, 32, 48]  # 学习率变化

## 测试模型名称
test_model_name = "1.pth"  # 待测试模型

## 保存路径
root_dir = os.path.abspath(os.path.dirname(__file__))
data_root = join(dirname(root_dir), "data_processing", "data")
save_root_dir = join(root_dir, 'result')


def train():
    model = GoogLeNet()  # 定义网络
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(model.parameters(),  # 定义优化器
                                lr=lr_,
                                weight_decay=weight_decay
                                )
    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        Dataset(root=data_root, split="train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True)

    # 自动调节学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=steppoints, gamma=0.1
    )

    '''开始训练'''

    # 存放训练、验证的损失和精度
    train_loss = []
    train_accuracy = []

    CUDA = torch.cuda.is_available()

    if CUDA:
        model.cuda()  # 移到GPU上运行

    for epoch in range(num_epochs):

        '''如果您需要显示loss图，请取消下面几行注释'''
        '''请安装pip install hiddenlayer '''
        # 存储评价指标
        # history = hl.History()
        # canvas1 = hl.Canvas()
        # canvas2 = hl.Canvas()

        start = time.time()

        lr_scheduler.step()
        correct = 0
        iter_loss = 0.0  # 累加loss
        model.train()  # 将网络放置训练模式

        for i, (inputs, labels, image_path) in enumerate(train_loader):

            inputs = Variable(inputs)  # 转为Variable变量
            labels = Variable(labels)

            if CUDA:
                inputs = inputs.float().cuda()  # 转为float格式并移到GPU
                labels = labels.long().cuda()

            optimizer.zero_grad()  # 清除上次运行时保存的梯度

            outputs = model(inputs)  # 输入模型得到输出

            loss = criterion(outputs, labels)  # 计算loss （对于一张图像，output是对于的猫和狗的概率，lable为正确的值）
            iter_loss += loss.item()  # 保存loss
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器优化网络参数

            _, predicted = torch.max(outputs, 1)  # 预测结果
            correct += (predicted == labels).sum()  # 累加预测正确的图像个数

            '''如果您需要显示loss图，请取消下面几行注释'''
            # 显示指标
            # if i % 5 == 0:
            #     # 存到评价指标到history对象
            #     history.log(i, loss=loss, accuracy=int(correct) / ((len(train_loader) * batch_size)))
            #     # 在图中绘制两个指标
            #     canvas1.draw_plot([history["loss"]])
            #     canvas2.draw_plot([history["accuracy"]])
            #     time.sleep(0.1)

        train_loss.append(iter_loss / (len(train_loader) * batch_size))  # 训练数据集中的loss
        train_accuracy.append((int(correct) * 100 / ((len(train_loader) * batch_size))))  # 训练数据集中的精度
        stop = time.time()
        lr = lr_scheduler.get_lr()[0]
        print('Epoch:{}/{} \t Trainin Loss:{:.6f} \t Training Accuracy:{:.6f}% \t Time:{:.3f}s \t lr:{}'.format(
            epoch + 1,
            num_epochs,
            train_loss[
                -1],
            train_accuracy[
                -1],
            stop - start, lr))

        # 保存模型
        save_dir = join(save_root_dir, exp_name)  # 文件夹以参数中的实验名称来命名
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model, save_dir + '/' + str(epoch) + '.pth')
        torch.save(model.state_dict(), save_dir + '/' + str(epoch) + "_dict" + '.pth')
        # 保存日志
        optStr = "model:{}\t optimizer:{}\t batch_size:{}\t lr:{}\t steppoints:{}\n".format("GoogleNet",
                                                                                            "SGD",
                                                                                            batch_size, lr,
                                                                                            steppoints)
        logfile = save_dir + "/{}.log".format(exp_name)
        logStr = optStr + "Epoch\tTraining Loss\tTraining Acc\tTime\n" if epoch == 0 else ""
        logStr += "{}\t{:.6f}\t{:.6f}\t{:.3f}\n".format(epoch + 1, train_loss[-1],
                                                        train_accuracy[-1], stop - start)

        with open(logfile, 'a') as fn:
            fn.write(logStr)


def test_model():
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    # 加载训练数据
    test_loader = torch.utils.data.DataLoader(
        Dataset(root=data_root, split="test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True)
    for epoch in range(num_epochs):
        model_name = "{}.pth".format(epoch)  # 模型名称

        model = torch.load(join(save_root_dir, exp_name, model_name))
        model.eval()  # Put the network into evaluation mode

        iter_loss = 0.0
        correct = 0.0

        start = time.time()
        # logStr = str(time.localtime(time.time())) + "\n"
        for i, (inputs, labels, image_path) in enumerate(test_loader):
            # Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculate the loss
            iter_loss += loss.item()
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()

        # Record the Testing loss
        test_loss = iter_loss / (len(test_loader) * batch_size)
        # Record the Testing accuracy
        test_accuracy = int(correct) * 100 / (len(test_loader) * batch_size)
        stop = time.time()

        # 保存日志
        save_dir = join(save_root_dir, exp_name)
        logfile = save_dir + "/{}.log".format(exp_name)

        logStr = 'Model:{} \t Testing Loss: {:.8f} \t Testing Acc: {:.8f} \t Total Time: {:.4f}s \t Test Num:{}\n'.format(
            model_name,
            test_loss,
            test_accuracy,
            stop - start,
            len(test_loader) * batch_size)
        print(logStr)
        with open(logfile, 'a') as fn:
            fn.write(logStr)


if __name__ == '__main__':
    train()
    # 测试前将batch_size改为1
    # test_model()
