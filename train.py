from datetime import datetime

import torch
from torch.utils import tensorboard
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from LeNet import LeNet5

# RESUME_PATH = None
RESUME_PATH = "./log/2022-08-15_21-22-40/LenetMnist9.pt"

# 定义超参数
EPOCHS = 50
LEARNINGRATE = 1e-2
BATCHSIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定训练设备

# 0.图片预处理
compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(28, 28)),
    transforms.Normalize([0.5], [0.5])  # 要根据通道数设置均值和方差
])

# 1.加载数据集
trainMnist = datasets.MNIST(root="./datasets", train=True, transform=compose, download=True)
testMnist = datasets.MNIST(root="./datasets", train=False, transform=compose, download=True)
# 数据集长度
trainLen = len(trainMnist)
testLen = len(testMnist)
print("训练测试集长度：", trainLen)
print("测试数据集长度：", testLen)

# 2.用downloader批量读取数据集
trainBatch = data.DataLoader(dataset=trainMnist, batch_size=BATCHSIZE, shuffle=True, drop_last=False)
testBatch = data.DataLoader(dataset=testMnist, batch_size=BATCHSIZE, shuffle=True, drop_last=False)

# 3.定义要使用的网络LeNet
leNet = LeNet5()
leNet.to(device)
# 4.定义损失函数函数（损失函数已经对minibatch求过平均）
lossFun = nn.CrossEntropyLoss()
lossFun.to(device)
# 5.定义优化器
sgdOptimizer = optim.SGD(leNet.parameters(), lr=LEARNINGRATE, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(sgdOptimizer, T_max=EPOCHS, eta_min=0.001)
# 判断是否继续训练
if RESUME_PATH:
    checkpoint = torch.load(RESUME_PATH)
    start_epoch = checkpoint["start_epoch"] + 1
    leNet.load_state_dict(checkpoint["model_state_dict"])
    sgdOptimizer.load_state_dict(checkpoint["Optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
else:
    start_epoch = 0
# 6.训练网络
# 当前每轮训练次数，一个minibitch为一轮
trainNum = 0
# 当前每轮测试次数，一个epoch为一轮
testNum = 0
# 创建tensorboar 来记录网络
time_str = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
writer = tensorboard.SummaryWriter("./log/" + time_str)
# 开始epoch循环
for epoch in range(start_epoch, EPOCHS):
    # 训练开始标志，能够开启batch-normalization和dropout
    batch_num_train = 0  # 一个epoch中batch 的数目
    train_loss_sum = 0
    for datas in trainBatch:
        # 获得数据
        imgs, labels = datas
        imgs = imgs.to(device)
        labels = labels.to(device)
        # 将数据输入
        y = leNet(imgs)
        # 计算loss
        loss = lossFun(y, labels)
        # 清空模型梯度
        sgdOptimizer.zero_grad()
        # 反向传播求导，更新模型梯度
        loss.backward()
        # 优化器更新模型的权重
        sgdOptimizer.step()
        train_loss_sum += loss
        batch_num_train += 1
    tain_loss_mean = train_loss_sum / batch_num_train
    writer.add_scalar("train/loss", scalar_value=tain_loss_mean, global_step=epoch)  # 记录每一百个bitch（640个）后的loss
    writer.add_scalar("learning_rate", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)

    # 更新学习率
    scheduler.step()

    # 测试步骤开始
    leNet.eval()
    test_correct_sum = 0  # 每一个epoch轮 总数据中正确识别标签的数量
    test_loss_sum = 0  # 每一个epoch轮 总数据/minibatch个loss的和
    batch_num_test = 0
    with torch.no_grad():
        for imgs, labels in testBatch:
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 获得网络输出
            y = leNet(imgs)
            # 计算精度
            max_idx_list = y.argmax(1)  # 参数为1可以按照【0】【1】每一行，返回一个一维的张量
            test_correct_sum += (max_idx_list == labels).sum()
            # 计算损失
            test_loss_sum = lossFun(y, labels) + test_loss_sum
            batch_num_test += 1
    accuracy = test_correct_sum / testLen
    test_loss_mean = test_loss_sum / batch_num_test  # 因为损失函数自动对每一个minibitch求平均，所以loss的和为每一个minibitch的数量
    print("第{0}次整体测试正确率{1}".format(epoch, accuracy))
    print("第{0}次整体测试损失{1}".format(epoch, test_loss_mean.item()))
    writer.add_scalar("test/acc", scalar_value=accuracy, global_step=epoch)
    writer.add_scalar("test/loss", scalar_value=test_loss_mean, global_step=epoch)
    # 保存多种数据以方便继续训练
    state = {
        "start_epoch": epoch,
        "model_state_dict": leNet.state_dict(),
        "Optimizer_state_dict": sgdOptimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, "./log/" + time_str + "/LenetMnist{0}.pt".format(epoch))
    print("模型已保存")
writer.close()
