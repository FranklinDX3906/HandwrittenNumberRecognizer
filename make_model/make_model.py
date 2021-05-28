# coding:utf-8
# 使用cnn训练手写数字

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from myCNN import MyCNN


# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ]
)

data_path = './MNIST_DATA_PyTorch'
# 获取数据集
train_data = torchvision.datasets.mnist.MNIST(
    data_path, train=True, transform=data_tf, download=False)
test_data = torchvision.datasets.mnist.MNIST(
    data_path, train=False, transform=data_tf, download=False)

train_data_size = len(train_data)
valid_data_size = len(test_data)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=100, shuffle=True)

print('train_data_size: ', train_data_size,
      '   valid_data_size: ', valid_data_size)


model = MyCNN()
print(model)

model = model.to('cuda:0')

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50  # 训练轮数

print('\nThis train has epohs: ', epochs, '\n\n')

record = []
best_acc = 0.0
best_epoch = 0

# loss_count = []
for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch + 1, epochs), ':')

    # model.train()  # 训练

    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda:0')
        labels = labels.to('cuda:0')  # 挂到GPU

        outputs = model.forward(inputs)  # 获得输出
        opt.zero_grad()  # 清空上一步残余庶出

        loss = loss_func(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        opt.step()  # 更新参数

        train_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)

        if(i % 100 == 0):
            print('{}:\t'.format(i), loss.item())

    with torch.no_grad():
        model.eval()

        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to('cuda:0')
            labels = labels.to('cuda:0')

            outputs = model(inputs)
            # outputs = outputs.cuda()

            # acc = torch.max(outputs.data, 1) == labels.data

            loss = loss_func(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(
                labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)
            avg_train_loss = train_loss / train_data_size

            # print('{}:\t'.format(i), valid_acc/)

    avg_train_acc = train_acc / train_data_size
    avg_valid_loss = valid_loss / valid_data_size
    avg_valid_acc = valid_acc / valid_data_size

    record.append([avg_train_loss, avg_valid_loss,
                   avg_train_acc, avg_valid_acc])

    if(avg_valid_acc > best_acc):  # 记录最高准确性的模型
        best_acc = avg_valid_acc
        best_epoch = epoch + 1

    epoch_end = time.time()

    print("Training: Loss: {:.4f}, \nAccuracy: {:.4f}%, \nValidation: Loss: {:.4f}, \nAccuracy: {:.4f}%, \nTime: {:.4f}s".format(
        avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start))
    print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(
        best_acc, best_epoch), '\n\n\n')


# 打印结果
record = np.array(record)
plt.plot(record[:, 0:2])
plt.legend(['Train Loss', 'Valid Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('loss.png')
plt.show()

plt.plot(record[:, 2:4])
plt.legend(['Train Accuracy', 'Valid Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('accuracy.png')
plt.show()

# 记录结果
torch.save(model, '../mycnn_minist.pth')
