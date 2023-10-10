import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride, ShortCutFlag):
        super(ResidualBlock, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=stride[0],
                            padding=1), torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=stride[1],
                            padding=1), torch.nn.BatchNorm2d(out_channels))
        self.shortcut = torch.nn.Sequential()
        self.dropout = torch.nn.Dropout(p=0.3)
        if ShortCutFlag == True:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=2),
                torch.nn.BatchNorm2d(2 * in_channels))

    def forward(self, x):
        output = self.layer(x)

        # output += self.shortcut(x)
        output = self.dropout(output) + self.shortcut(x)

        output = F.relu(output)
        return output


class ResNet18(torch.nn.Module):

    def __init__(self, ResidualBlock):
        super(ResNet18, self).__init__()

        self.stage1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=64,
                            kernel_size=7,
                            stride=2,
                            padding=3), torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage2 = torch.nn.Sequential(
            ResidualBlock(in_channels=64,
                          out_channels=64,
                          stride=[1, 1],
                          ShortCutFlag=False))
        self.stage3 = torch.nn.Sequential(
            ResidualBlock(in_channels=64,
                          out_channels=64,
                          stride=[1, 1],
                          ShortCutFlag=False))
        self.stage4 = torch.nn.Sequential(
            ResidualBlock(in_channels=64,
                          out_channels=128,
                          stride=[2, 1],
                          ShortCutFlag=True))
        self.stage5 = torch.nn.Sequential(
            ResidualBlock(in_channels=128,
                          out_channels=128,
                          stride=[1, 1],
                          ShortCutFlag=False))
        self.stage6 = torch.nn.Sequential(
            ResidualBlock(in_channels=128,
                          out_channels=256,
                          stride=[2, 1],
                          ShortCutFlag=True))
        self.stage7 = torch.nn.Sequential(
            ResidualBlock(in_channels=256,
                          out_channels=256,
                          stride=[1, 1],
                          ShortCutFlag=False))
        self.stage8 = torch.nn.Sequential(
            ResidualBlock(in_channels=256,
                          out_channels=512,
                          stride=[2, 1],
                          ShortCutFlag=True))
        self.stage9 = torch.nn.Sequential(
            ResidualBlock(in_channels=512,
                          out_channels=512,
                          stride=[1, 1],
                          ShortCutFlag=False))
        self.stage10 = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = torch.nn.Linear(512, 500)

    def forward(self, x):
        output = self.stage1(x)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.stage5(output)
        output = self.stage6(output)
        output = self.stage7(output)
        output = self.stage8(output)
        output = self.stage9(output)
        output = self.stage10(output)

        output = output.reshape(x.shape[0], -1)
        output = self.fc(output)

        return output


def train(network, train_loader, optimizer, loss, device):
    network.train()
    for data, target in train_loader:
        #print(target.shape[0])
        #print(loss_count)
        labels = torch.zeros(target.shape[0], 500)
        for i in range(target.shape[0]):
            labels[i][target[i]] = 1

        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        output = network(data).squeeze()

        l = loss(output, labels)

        l.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=network.parameters(),
        #                                max_norm=20,
        #                                norm_type=2)
        optimizer.step()


def Read_Data(root, data_transforms, type):
    # ImageFolder 通用的加载器
    data = torchvision.datasets.ImageFolder(root,
                                            transform=data_transforms[type])
    #print(train_data)
    return data


def accuracy(epoch_idx, test_loader, network, device, loss, set_type=None):
    correct = 0
    loss_count = 0
    if (set_type == 'val'):
        network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            labels = torch.zeros(target.shape[0], 500)
            for i in range(target.shape[0]):
                labels[i][target[i]] = 1
            labels = labels.to(device)
            outputs = network(data)
            l = loss(outputs, labels)
            loss_count += (l.item() * target.shape[0])
            Pred, PredIndex = torch.max(outputs.data, dim=1)
            correct += (int((PredIndex == target).sum()))

    if set_type == "train":
        print('\nEpoch{}: Train accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    if set_type == "val":
        print('\nEpoch{}: Test accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset), loss_count / len(
        test_loader.dataset)


# 构建可迭代的数据装载器
if __name__ == "__main__":
    train_root = './face_classification_500/train_sample'
    validation_root = "./face_classification_500/dev_sample"
    test_root = "./face_classification_500/test_sample"
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    learing_rate = 0.1
    epoches = 55
    batch_size = 32
    train_data = Read_Data(root=train_root,
                           data_transforms=data_transforms,
                           type='train')
    validation_data = Read_Data(root=validation_root,
                                data_transforms=data_transforms,
                                type='val')
    test_data = Read_Data(root=test_root,
                          data_transforms=data_transforms,
                          type='val')
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    validation_loader = DataLoader(dataset=validation_data,
                                   batch_size=1,
                                   shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    loss_fun = torch.nn.CrossEntropyLoss()
    network = ResNet18(ResidualBlock).cuda()  #to(device)
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learing_rate,
                                momentum=0.8)
    #print(network)
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    epoch = []
    for i in range(1, epoches + 1):
        epoch.append(i)
        print(f"Epoch {i}\n-------------------------------")
        curr_loss = train(train_loader=train_loader,
                          optimizer=optimizer,
                          network=network,
                          loss=loss_fun,
                          device=device)
        #if i % 10 == 0:
        train_accuracy, curr_loss = accuracy(epoch_idx=i,
                                             test_loader=train_loader,
                                             network=network,
                                             loss=loss_fun,
                                             device=device,
                                             set_type="train")
        train_loss.append(curr_loss)
        print(curr_loss)
        val_accuracy, curr_loss = accuracy(epoch_idx=i,
                                           test_loader=validation_loader,
                                           network=network,
                                           loss=loss_fun,
                                           device=device,
                                           set_type="val")
        val_loss.append(curr_loss)
        print(curr_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

    torch.save(network, 'net10086.pth')

    # 测试模型在测试集上表现的代码
    # network = torch.load('net10086.pth')
    # print(
    #     accuracy(epoch_idx=1,
    #              test_loader=test_loader,
    #              network=network,
    #              loss=loss_fun,
    #              device=device,
    #              set_type="val"))

    print(train_loss)
    print(train_acc)
    print(val_acc)
    print(val_loss)

    plt.figure()
    plt.title('loss/acc')
    plt.xlabel("iterations")
    plt.ylabel("loss/acc")

    plt.plot(epoch, train_loss, epoch, train_acc, epoch, val_acc, epoch,
             val_loss)
    plt.show()