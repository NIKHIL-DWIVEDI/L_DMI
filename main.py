# main - training and testing for Fashion-MNIST

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from fashion import *

num_classes = 2
r=0.6
CE = nn.CrossEntropyLoss().cuda()




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out


def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    mat = mat / target.size(0)
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def train(train_loader, model, optimizer, criterion=CE):

    model.train()

    for i, (idx, input, target) in enumerate(tqdm(train_loader)):
        if idx.size(0) != batch_size:
            break

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for i, (idx, input, target) in enumerate(tqdm(test_loader)):
        input = torch.Tensor(input).cuda()
        target = torch.autograd.Variable(target).cuda()

        total += target.size(0)
        output = model(input)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def validate(valid_loader, model, criterion):

    model.eval()
    loss_t = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(valid_loader):
            if i == 1:
                break

            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            output = model(input)
            loss = criterion(output, target)
            loss_t += loss

    print('valid_loss=', loss_t.item())
    return loss_t


def main_ce():
    model_ce = CNNModel().cuda()
    best_valid_acc = 0

    for epoch in range(20):
        print("epoch=", epoch)
        print("r=", r)
        learning_rate = 1e-4
        optimizer_ce = torch.optim.Adam(model_ce.parameters(), lr=learning_rate)
        print("traning model_ce...")
        train(train_loader=train_loader, model=model_ce, optimizer=optimizer_ce)
        print("validating model_ce...")
        valid_acc = test(model=model_ce, test_loader=valid_loader)
        print('valid_acc=', valid_acc)
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model_ce, './model_ce_' + str(r) + '_' + str(s) + '_' + str(c))
            print("saved.")
        print("testing model_ce...")
        test_acc = test(model=model_ce, test_loader=test_loader)
        print('test_acc=', test_acc)


def main_dmi():
    model_dmi = torch.load('./model_ce_' + str(r) + '_' + str(s) + '_' + str(c))
    test_acc = test(model=model_dmi, test_loader=test_loader)
    print('test_acc=', test_acc)
    best_valid_loss = validate(valid_loader=valid_loader, model=model_dmi, criterion=DMI_loss)

    for epoch in range(20):
        print("epoch=", epoch)
        print('r=', r)
        learning_rate = 1e-4
        optimizer_dmi = torch.optim.Adam(model_dmi.parameters(), lr=learning_rate)

        print("traning model_dmi...")
        train(train_loader=train_loader, model=model_dmi, optimizer=optimizer_dmi, criterion=DMI_loss)
        print("validating model_dmi...")
        valid_loss = validate(valid_loader=valid_loader, model=model_dmi, criterion=DMI_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_dmi, './model_dmi_' + str(r) + '_' + str(s) + '_' + str(c))
            print("saved.")

def evaluate(path):
    model_dmi = torch.load(path + str(r) + '_' + str(s) + '_' + str(c))
    test_acc = test(model=model_dmi, test_loader=test_loader)
    print('final_test_acc=', test_acc)


# print("Hello")
main_ce()
main_dmi()
evaluate('./model_ce_')
evaluate('./model_dmi_')