# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author axiumao
"""

import os
import csv
import argparse
import time
import math
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
from torch.cuda.amp import autocast, GradScaler

from conf import settings
from Regularization import Regularization
from utils import get_network, get_mydataloader, get_weighted_mydataloader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score



def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls):
    scaler = GradScaler()
    start = time.time()
    network.train()
    train_acc_process = []
    train_loss_process = []

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast():
            outputs = network(images)
            # 定义loss_type
            loss_type = "focal"
            loss_cb = CB_loss(labels, outputs, samples_per_cls, 5, loss_type, args.beta, args.gamma)
            loss_ce = loss_function(outputs, labels)
            loss = 0.3 * loss_ce + 0.7 * loss_cb
            
            if args.weight_d > 0:
                loss += reg_loss(network)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        _, preds = outputs.max(1)
        correct_n = preds.eq(labels).sum()
        accuracy_iter = correct_n.float() / len(labels)

        # Move accuracy_iter to CPU for storing
        train_acc_process.append(accuracy_iter.cpu().numpy().tolist())
        train_loss_process.append(loss.item())

    # 打印信息时不用设备移动
    print('Training Epoch: {epoch} [{total_samples}]\tTrain_accuracy: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            np.mean(train_acc_process),
            np.mean(train_loss_process),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            total_samples=len(train_loader.dataset)
    ))

    mean_acc = np.mean(train_acc_process)
    mean_loss = np.mean(train_loss_process)
    
    # 返回训练指标
    return network, mean_acc, mean_loss



@torch.no_grad()
def eval_training(valid_loader, network, loss_function, epoch=0):
    start = time.time()
    network.eval()

    n = 0
    valid_loss = 0.0
    correct = 0.0
    class_target = []
    class_predict = []

    # 在 GPU 上收集所有数据，减少频繁的 GPU 到 CPU 转换
    for (images, labels) in valid_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = network(images)
        loss = loss_function(outputs, labels)

        valid_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        # 收集 GPU 上的张量
        class_target.append(labels)
        class_predict.append(preds)

        n += 1

    # 在所有数据收集完之后再进行转换到 CPU
    class_target = torch.cat(class_target).cpu().numpy().tolist()
    class_predict = torch.cat(class_predict).cpu().numpy().tolist()

    # 打印分类报告
    report = classification_report(class_target, class_predict, zero_division=0)
    print('Evaluating Network.....')
    print('Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
        epoch,
        valid_loss / n,
        correct.float() / len(valid_loader.dataset),
        time.time() - start
    ))
    print('Classification Report:')
    print(report)

    accuracy = correct.float() / len(valid_loader.dataset)
    avg_loss = valid_loss / len(valid_loader.dataset)
    f1 = f1_score(class_target, class_predict, average='macro', zero_division=0)
    
    # 确保返回的是CPU张量
    return accuracy.cpu(), avg_loss, f1

        

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST', help='net type')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoches')
    parser.add_argument('--seed',type=int, default=10, help='seed')
    parser.add_argument('--gamma',type=float, default=2.5, help='the gamma of focal loss')
    parser.add_argument('--beta',type=float, default=0.99, help='the beta of class balanced loss')
    parser.add_argument('--weight_d',type=float, default=0.1, help='weight decay for regularization')  # 权重衰减 系数 
    parser.add_argument('--save_path',type=str, default='setting0', help='saved path of each setting') #
    parser.add_argument('--data_path',type=str, default='C:\\Users\\10025\\Desktop\\0000PatchTST-TFC-main\\0000PatchTST-TFC-main\\CMI-Net\\data\\new_goat_25hz_3axis.pt', help='saved path of input data')
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
    
    train_loader, weight_train, number_train = get_weighted_mydataloader(pathway, data_id=0, batch_size=args.b, num_workers=num_workers, shuffle=True)
    valid_loader = get_mydataloader(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)
    
    # 将 weight_train 移动到正确的设备上
    if isinstance(weight_train, torch.Tensor):
        weight_train = weight_train.to(device)
    
    # 确保 number_train 也在正确的设备上
    if isinstance(number_train, torch.Tensor):
        number_train = number_train.to(device)
    
    if args.weight_d > 0:
        reg_loss=Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")
    
    # 使用标签平滑的交叉熵损失
    loss_function_CE = nn.CrossEntropyLoss(
        weight=weight_train,
        label_smoothing=0.1  # 添加标签平滑
    )
    optimizer = optim.Adam(
        net.parameters(), 
        lr=args.lr,
        weight_decay=0.01  # 增加权重衰减
    )

    # 添加学习率预热和余弦退火
    warmup_epochs = 5
    def warmup_cosine_schedule(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epoch - warmup_epochs)))
    
    # 使用新的学习率调度器
    train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):               # 如果没 log 路径 创建log路径
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
        # 训练并获取指标
        net, train_acc, train_loss = train(train_loader, net, optimizer, epoch, 
                                         loss_function=loss_function_CE, 
                                         samples_per_cls=number_train)
        
        # 验证并获取指标
        valid_acc, valid_loss, fs_valid = eval_training(valid_loader, net, loss_function_CE, epoch)
        
        # 记录所有指标 - 确保都是在CPU上
        Train_Accuracy.append(float(train_acc))  # 转换为Python float
        Train_Loss.append(float(train_loss))
        Valid_Accuracy.append(float(valid_acc))  # 转换为Python float
        Valid_Loss.append(float(valid_loss))
        f1_s.append(float(fs_valid))
        
        # 打印当前epoch的训练情况
        print('Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | F1: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, fs_valid
        ))
        
        train_scheduler.step()
        
        # 保存最佳模型
        if epoch > settings.MILESTONES[0] and best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)
            
    print('Best epoch: {} with accuracy: {:.4f}'.format(best_epoch, best_acc))


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
        
        precision_test = precision_score(test_target, test_predict, average='macro', zero_division=0)
        print('precision = {:.5f}'.format(precision_test), file=f)
        
        recall_test = recall_score(test_target, test_predict, average='macro', zero_division=0)
        print('recall = {:.5f}'.format(recall_test), file=f)
        
        #Output the classification report
        print('------------', file=f)
        print('Classification Report', file=f)
        print(classification_report(test_target, test_predict, zero_division=0), file=f)
        
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

    # 可选：添加dropout
    net.apply(lambda m: setattr(m, 'dropout', nn.Dropout(p=0.2)) if hasattr(m, 'dropout') else None)
