# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author axiumao
"""

import os
import csv
import argparse
import time
import numpy as np
import platform
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
# import torch.backends.cudnn as cudnn
# import matplotlib.font_manager as font_manager

import torch
import torch.nn as nn
import torch.optim as optim
from Class_balanced_loss import CB_loss

from conf import settings
from Regularization import Regularization
from utils import get_network, get_mydataloader, get_weighted_mydataloader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score



def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls):
    """
    训练函数，用于训练神经网络模型
    参数:
        train_loader: 训练数据加载器
        network: 神经网络模型
        optimizer: 优化器
        epoch: 当前训练轮次
        loss_function: 损失函数
        samples_per_cls: 每个类别的样本数量
    """
    # 记录开始时间
    start = time.time()
    # 将网络设置为训练模式
    network.train()
    # 初始化列表用于存储训练过程中的准确率和损失值
    train_acc_process = []
    train_loss_process = []

    # 遍历训练数据批次
    for batch_index, (images, labels) in enumerate(train_loader):
        # 将图像和标签数据移动到指定设备(GPU/CPU)
        images, labels = images.to(device), labels.to(device)

        # 清除之前的梯度
        optimizer.zero_grad()
        # 前向传播，获取模型输出
        outputs = network(images)

        # 确保samples_per_cls是张量并移动到正确的设备
        if isinstance(samples_per_cls, torch.Tensor):
            samples_per_cls = samples_per_cls.to(device)

        # 设置损失类型为focal loss
        loss_type = "focal"
        # 计算类别平衡损失(Class Balanced Loss)
        loss_cb = CB_loss(labels, outputs, samples_per_cls, 5, loss_type, args.beta, args.gamma)

        # 计算交叉熵损失
        loss_ce = loss_function(outputs, labels)
        # 组合损失(这里CB loss的权重为0，实际上只使用了CE loss)
        # loss = 1.0 * loss_ce + 0.0 * loss_cb
        loss = 0.0*loss_ce + 1.0*loss_cb # class-balanced focal loss (CMI-Net+CB focal loss)
        
        # 如果启用权重衰减，添加正则化损失
        if args.weight_d > 0:
            loss += reg_loss(network)

        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        # 计算预测准确率
        _, preds = outputs.max(1)  # 获取最大概率的类别
        correct_n = preds.eq(labels).sum()  # 计算正确预测的数量
        accuracy_iter = correct_n.float() / len(labels)  # 计算准确率

        # 将准确率和损失值存储到CPU中
        train_acc_process.append(accuracy_iter.cpu().numpy().tolist())
        train_loss_process.append(loss.item())

    # 打印训练信息
    print('Training Epoch: {epoch} [{total_samples}]\tTrain_accuracy: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            np.mean(train_acc_process),  # 平均准确率
            np.mean(train_loss_process),  # 平均损失
            optimizer.param_groups[0]['lr'],  # 当前学习率
            epoch=epoch,
            total_samples=len(train_loader.dataset)  # 总样本数
    ))

    # 计算并存储每个 epoch 的平均准确率 
    Train_Accuracy.append(np.mean(train_acc_process))
    Train_Loss.append(np.mean(train_loss_process))
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    return network



@torch.no_grad()  # 装饰器，评估时不需要更新梯度
def eval_training(valid_loader, network, loss_function, epoch=0):
    start = time.time()  # 记录开始时间
    network.eval()  # 将网络设置为评估模式，关闭dropout等训练特有的层

    n = 0  # 批次计数器
    valid_loss = 0.0  # 累计验证损失
    correct = 0.0  # 累计正确预测数
    class_target = []  # 存储真实标签
    class_predict = []  # 存储预测标签
    loss_function = loss_function.cuda()

    # 在 GPU 上收集所有数据，减少频繁的 GPU 到 CPU 转换
    for (images, labels) in valid_loader:  # 遍历验证数据集
        # images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备(GPU/CPU)  cuda:0

        images = images.cuda()
        labels = labels.cuda()
            

        outputs = network(images)  # 前向传播得到模型输出 [batch_size, 概率值]
        loss = loss_function(outputs, labels)  # 计算损失值

        valid_loss += loss.item()  # 累加损失值
        _, preds = outputs.max(1)  # 获取最大概率对应的类别作为预测结果
        correct += preds.eq(labels).sum()  # 计算预测正确的样本数

        # print(f"预测preds: {preds},预测长度,{len(preds)}, 真实labels: {labels},真实长度,{len(labels)}")  batch_size 256
        
        # 收集 GPU 上的张量
        class_target.append(labels)  # 收集真实标签
        class_predict.append(preds)  # 收集预测标签

        n += 1  # 批次计数加1

    # 在所有数据收集完之后再进行转换到 CPU
    class_target = torch.cat(class_target).cpu().numpy().tolist()  # 将所有真实标签合并并转换为列表
    class_predict = torch.cat(class_predict).cpu().numpy().tolist()  # 将所有预测标签合并并转换为列表


    print('Evaluating Network.....')
    print('Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
        epoch,
        valid_loss / n,  # 计算平均损失
        correct.float() / len(valid_loader.dataset),  # 计算准确率
        time.time() - start  # 计算耗时
    ))

    # 生成混淆矩阵
    matrix = confusion_matrix(class_target, class_predict)
    print('Confusion Matrix:')
    print(matrix)

    
    # # 打印分类报告
    # report = classification_report(class_target, class_predict, zero_division='warn')  # 生成分类报告
    # print('------------')
    # print('Classification Report:')
    # print(report)

    #Obtain f1_score of the prediction
    fs = f1_score(class_target, class_predict, average='macro')
    print('f1 score = {}'.format(fs))
    f1_s.append(fs)
    Valid_Loss.append(valid_loss / n)
    Valid_Accuracy.append(correct.float() / len(valid_loader.dataset))
    

    # 返回准确率、平均损失和宏平均F1分数
    return correct.float() / len(valid_loader.dataset), valid_loss / len(valid_loader.dataset), f1_score(class_target, class_predict, average='macro')

        

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST', help='net type')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epoch',type=int, default=100, help='total training epoches')
    parser.add_argument('--seed',type=int, default=10, help='seed')
    parser.add_argument('--gamma',type=float, default=0, help='the gamma of focal loss')
    parser.add_argument('--beta',type=float, default=0.9999, help='the beta of class balanced loss')
    parser.add_argument('--weight_d',type=float, default=0.1, help='weight decay for regularization')  # 权重衰减 系数 
    parser.add_argument('--save_path',type=str, default='setting0', help='saved path of each setting') #
    parser.add_argument('--data_path',type=str, default='E:\program\aaa_DL_project\0000PatchTST-TFC\CMI-Net\data\new_goat_25hz_3axis.pt', help='saved path of input data')
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.gpu > 0 and torch.cuda.is_available() else "cpu") # 条件运算符，如果 args.gpu > 0 并且 torch.cuda.is_available() 为 True，则使用 GPU，否则使用 CPU

    if args.gpu:
        torch.cuda.manual_seed(args.seed)# 设置 GPU 上的随机数种子，确保在 GPU 上的随机操作（如权重初始化等）也是可重复的
    else:
        torch.manual_seed(args.seed)#  设置 CPU 上的随机数种子，确保在 CPU 上执行的所有与随机性相关的操作都是可重复的
    
    net = get_network(args).to(device)   # get_network 在 utils.py  中 ，把模型搬运到device(GPU)中
    # print(net)
    print(f"Model is on device: {next(net.parameters()).device}")
    # print(f"Model is on device: {net.parameters().device}")
    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(args.epoch, args.b, args.lr, args.gpu, args.seed))

    sysstr = platform.system()
    if(sysstr =="Windows"):
        num_workers = 0
    else:
        num_workers = 8                        # 在windows上的进程是0， 在Linux的是8？ 在Windows 在多进程的数据加载时可能会遇到问题？？？？
        
    pathway = args.data_path                     # 默认Linux的问题
    if sysstr=='Linux': 
        pathway = args.data_path
    
    train_loader, weight_train, number_train = get_weighted_mydataloader(pathway, data_id=0, batch_size=args.b, num_workers=num_workers, shuffle=True) # weight_train 是每个类别的权重 ，number_train 是每个类别的样本数量
    valid_loader = get_mydataloader(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)
    
    if args.weight_d > 0:
        reg_loss=Regularization(net, args.weight_d, p=2) # 正则化损失函数
    else:
        print("no regularization")
    
    # loss_function = nn.CrossEntropyLoss(weight=weight_train)
    loss_function_CE = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=args.lr) # 使用 Adam 优化器来训练模型，并指定学习率 args.lr
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    #  创建一个学习率调度器 StepLR，每 20 个 epoch 调整学习率，缩小比例 gamma=0.1


    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path, settings.TIME_NOW) #它会根据操作系统的路径分隔符（例如，Windows 上是反斜杠 \，而在 Unix/Linux 上是正斜杠 /）来正确地构建路径。
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):               # 如果没 log 路径 创建log路径 runs 路径
        os.mkdir(settings.LOG_DIR)                       
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):                  # 参数路径
        os.makedirs(checkpoint_path)
    checkpoint_path_pth = os.path.join(checkpoint_path, '{net}-{type}.pth')


    best_acc = 0.0
    Train_Loss = []
    Train_Accuracy = []
    Valid_Loss = []
    Valid_Accuracy = []
    f1_s = []
    best_epoch = 1
    best_weights_path = checkpoint_path_pth.format(net=args.net, type='best')
   
   
    # validation_loss = 0
    for epoch in range(1, args.epoch + 1):
        net = train(train_loader, net, optimizer, epoch, loss_function=loss_function_CE, samples_per_cls=number_train)
        acc, validation_loss, fs_valid = eval_training(valid_loader, net, loss_function_CE, epoch)
        train_scheduler.step()  # 去掉 epoch 参数，并放到最后调用

        #start to save best performance model (according to the accuracy on validation dataset) after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0] and best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)
    print('best epoch is {}'.format(best_epoch))


    #plot accuracy varying over time
    font_1 = {'weight' : 'normal', 'size'   : 20}
    fig1=plt.figure(figsize=(12,9))
    plt.title('Accuracy',font_1)
    index_train = list(range(1,len(Train_Accuracy)+1))
    plt.plot(index_train,Train_Accuracy,color='skyblue',label='train_accuracy')
    plt.plot(index_train,Valid_Accuracy,color='red',label='valid_accuracy')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Accuracy',font_1)
    
    acc_figuresavedpath = os.path.join(checkpoint_path,'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath)
    # plt.show()
    
    #plot loss varying over time
    fig2=plt.figure(figsize=(12,9))
    plt.title('Loss',font_1)
    index_valid = list(range(1,len(Valid_Loss)+1))
    plt.plot(index_valid,Train_Loss,color='skyblue', label='train_loss')
    plt.plot(index_valid,Valid_Loss,color='red', label='valid_loss')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    loss_figuresavedpath = os.path.join(checkpoint_path,'Loss_curve.png')
    plt.savefig(loss_figuresavedpath)
    # plt.show()
    
    #plot f1 score varying over time
    fig3=plt.figure(figsize=(12,9))
    plt.title('F1-score',font_1)
    index_fs = list(range(1,len(f1_s)+1))
    plt.plot(index_fs,f1_s,color='skyblue')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    fs_figuresavedpath = os.path.join(checkpoint_path,'F1-score.png')
    plt.savefig(fs_figuresavedpath)
    # plt.show()
    
    out_txtsavedpath = os.path.join(checkpoint_path,'output.txt')
    f = open(out_txtsavedpath, 'w+')
    
    print('Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Data path: {}, Saved path: {}'.format(
        args.seed, args.epoch, args.b, args.lr, args.weight_d, args.gpu, args.data_path, args.save_path),
        file=f)
    
    print('index: {}; maximum value of validation accuracy: {}.'.format(Valid_Accuracy.index(max(Valid_Accuracy))+1, max(Valid_Accuracy)), file=f)
    print('index: {}; maximum value of validation f1-score: {}.'.format(f1_s.index(max(f1_s))+1, max(f1_s)), file=f)
    print('--------------------------------------------------')
    print('Validation accuracy: {}'.format(Valid_Accuracy), file=f)
    print('Validation F1-score: {}'.format(f1_s), file=f)
    
    ######load the best trained model and test testing data  ，测试函数，推理
    best_net = get_network(args)
    best_net.load_state_dict(torch.load(best_weights_path))
    
    total_num_paras, trainable_num_paras = get_parameter_number(best_net)
    print('The total number of network parameters = {}'.format(total_num_paras), file=f)
    print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)
    
    best_net.eval()
    number = 0
    correct_test = 0.0
    test_target =[]
    test_predict = []
    
    with torch.no_grad():
        
        start = time.time()
        
        for n_iter, (image, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                labels = labels.cuda()

            output = best_net(image)
            output = torch.softmax(output, dim= 1)
            preds = torch.argmax(output, dim =1)
            # _, preds = output.topk(5, 1, largest=True, sorted=True)
            # _, preds = output.max(1)
            correct_test += preds.eq(labels).sum()
            
            if args.gpu:
                labels = labels.cpu()
                preds = preds.cpu()
        
            test_target.extend(labels.numpy().tolist())
            test_predict.extend(preds.numpy().tolist())
        
            number +=1
        
        print('Label values: {}'.format(test_target), file=f)
        print('Predicted values: {}'.format(test_predict), file=f)

        finish = time.time()
        accuracy_test = correct_test.float() / len(test_loader.dataset)
        print('Testing network......', file=f)
        print('Test set: Accuracy: {:.5f}, Time consumed: {:.5f}s'.format(
            accuracy_test,
            finish - start
            ), file=f)
        
        #Obtain f1_score of the prediction
        fs_test = f1_score(test_target, test_predict, average='macro')
        print('f1 score = {:.5f}'.format(fs_test), file=f)
        
        kappa_value = cohen_kappa_score(test_target, test_predict)
        print("kappa value = {:.5f}".format(kappa_value), file=f)
        
        precision_test = precision_score(test_target, test_predict, average='macro')
        print('precision = {:.5f}'.format(precision_test), file=f)
        
        recall_test = recall_score(test_target, test_predict, average='macro')
        print('recall = {:.5f}'.format(recall_test), file=f)
        
        #Output the classification report
        print('------------', file=f)
        print('Classification Report', file=f)
        print(classification_report(test_target, test_predict), file=f)
        
        if not os.path.exists('./results.csv'):
            with open("./results.csv", 'w+') as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(['index','accuracy','f1-score','precision','recall','kappa','time_consumed'])
        
        with open("./results.csv", 'a+') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([args.seed, accuracy_test, fs_test, precision_test, recall_test, kappa_value, finish-start])
        
        Class_labels = ['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
        #Show the confusion matrix so that it can help us observe the results more intuitively
        def show_confusion_matrix(validations, predictions):
            matrix = confusion_matrix(validations, predictions) #No one-hot
            #matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix,
                  cmap="coolwarm",
                  linecolor='white',
                  linewidths=1,
                  xticklabels=Class_labels,
                  yticklabels=Class_labels,
                  annot=True,
                  fmt="d")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_figuresavedpath = os.path.join(checkpoint_path,'Confusion_matrix.png')
            plt.savefig(cm_figuresavedpath)

        show_confusion_matrix(test_target, test_predict)
    
    if args.gpu:
        print('GPU INFO.....', file=f)
        print(torch.cuda.memory_summary(), end='', file=f)
