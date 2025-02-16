# train.py
# !/usr/bin/env	python3

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
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from conf import settings
from Regularization import Regularization
from utils import get_network, get_mydataloader, get_weighted_mydataloader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, \
    precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from models.PatchTST_wyh import PatchTSTNet  # 添加这行

# 在文件开头，主函数之前添加全局变量
global_vars = {
    'learning_rates': [],
    'class_accuracies': {},
    'class_f1_scores': {},
}

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA执行以获得更好的错误信息

def balance_batch(images, labels):
    """平衡每个batch中的类别"""
    device = images.device
    
    # 获取输入张量的维度
    input_shape = images.shape
    num_dims = len(input_shape)
    
    classes, class_counts = torch.unique(labels, return_counts=True)
    max_count = class_counts.max().item()
    
    balanced_images = []
    balanced_labels = []
    
    for cls in classes:
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        cls_labels = labels[cls_mask]
        
        if len(cls_images) < max_count:
            repeat_times = max_count // len(cls_images)
            remainder = max_count % len(cls_images)
            
            # 创建正确的重复维度
            repeat_dims = [repeat_times] + [1] * (num_dims - 1)
            cls_images = cls_images.repeat(*repeat_dims)
            cls_labels = cls_labels.repeat(repeat_times)
            
            if remainder > 0:
                idx = torch.randperm(len(cls_images))[:remainder]
                cls_images = torch.cat([cls_images, cls_images[idx]])
                cls_labels = torch.cat([cls_labels, cls_labels[idx]])
        
        balanced_images.append(cls_images)
        balanced_labels.append(cls_labels)
    
    balanced_images = torch.cat(balanced_images)
    balanced_labels = torch.cat(balanced_labels)
    
    # 随机打乱
    idx = torch.randperm(len(balanced_images))
    balanced_images = balanced_images[idx]
    balanced_labels = balanced_labels[idx]
    
    return balanced_images.to(device), balanced_labels.to(device)


def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls):
    global global_vars
    start = time.time()
    network.train()
    train_acc_process = []
    train_loss_process = []

    # 检查samples_per_cls的设备位置和类型
    if isinstance(samples_per_cls, np.ndarray):
        samples_per_cls = torch.from_numpy(samples_per_cls).float()
    samples_per_cls = samples_per_cls.to(device)

    # 在训练循环中添加每个类别的预测统计
    num_classes = 6  # 修改这里
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)

    # 记录学习率
    global_vars['learning_rates'].append(optimizer.param_groups[0]['lr'])

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images, labels = balance_batch(images, labels)

        optimizer.zero_grad()
        outputs = network(images)

        loss_type = "focal"
        loss_cb = CB_loss(labels, outputs, samples_per_cls, num_classes, loss_type, args.beta, args.gamma)
        loss_ce = loss_function(outputs, labels)
        loss = 1.0 * loss_ce + 0.0 * loss_cb

        if args.weight_d > 0:
            loss += reg_loss(network)

        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        correct_n = preds.eq(labels).sum()
        accuracy_iter = correct_n.float() / len(labels)

        # Move accuracy_iter to CPU for storing
        train_acc_process.append(accuracy_iter.cpu().numpy().tolist())
        train_loss_process.append(loss.item())

        # 统计每个类别的预测情况
        for label in range(num_classes):
            mask = labels == label
            class_correct[label] += (preds[mask] == labels[mask]).sum().item()
            class_total[label] += mask.sum().item()

    # 打印信息时不用设备移动
    print('Training Epoch: {epoch} [{total_samples}]\tTrain_accuracy: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        np.mean(train_acc_process),
        np.mean(train_loss_process),
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        total_samples=len(train_loader.dataset)
    ))

    # 打印每个类别的准确率
    for i in range(num_classes):
        print(f'Class {i} accuracy: {100 * class_correct[i] / class_total[i]:.2f}%')

    return network


@torch.no_grad()
def eval_training(valid_loader, network, loss_function, epoch=0):
    global global_vars
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

    # 计算每个类别的F1分数
    class_f1 = f1_score(class_target, class_predict, average=None)
    
    # 记录每个类别的F1分数
    for i, f1 in enumerate(class_f1):
        if i not in global_vars['class_f1_scores']:
            global_vars['class_f1_scores'][i] = []
        global_vars['class_f1_scores'][i].append(f1)

    return correct.float() / len(valid_loader.dataset), valid_loss / len(valid_loader.dataset), f1_score(class_target, class_predict, average='macro')


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST_wyh', help='net type')
    parser.add_argument('--gpu', type=int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoches')
    parser.add_argument('--seed', type=int, default=10, help='seed')
    parser.add_argument('--gamma', type=float, default=0, help='the gamma of focal loss')
    parser.add_argument('--beta', type=float, default=0.9999, help='the beta of class balanced loss')
    parser.add_argument('--weight_d', type=float, default=0.1, help='weight decay for regularization')  # 权重衰减 系数
    parser.add_argument('--save_path', type=str, default='setting0', help='saved path of each setting')
    parser.add_argument('--data_path', type=str, default='C:\\Users\\10025\\Desktop\\CMI-Net\\myTensor_Gyr_6.pt',
                        help='saved path of input data')
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.gpu > 0 and torch.cuda.is_available() else "cpu")

    if args.gpu:
        torch.cuda.manual_seed(args.seed)  # 设置 GPU 上的随机数种子，确保在 GPU 上的随机操作（如权重初始化等）也是可重复的
    else:
        torch.manual_seed(args.seed)  # 设置 CPU 上的随机数种子，确保在 CPU 上执行的所有与随机性相关的操作都是可重复的

    # 在创建模型之前先定义类别数
    Class_labels = ['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
    num_classes = len(Class_labels)  # 定义类别数
    print(f"Initializing model with {num_classes} classes")

    # 然后创建模型
    net = get_network(args).to(device)
    
    # 添加模型结构检查
    print("Model structure:")
    print(net)
    
    # 检查最后一层的输出维度
    last_layer = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            last_layer = module
    if last_layer is not None:
        print(f"Last layer output dimension: {last_layer.out_features}")
        if last_layer.out_features != num_classes:
            # 动态修改输出层
            if isinstance(net, PatchTSTNet):
                in_features = last_layer.in_features
                net.classifier = nn.Linear(in_features, num_classes).to(device)
                print(f"Modified classifier output dimension to {num_classes}")
            else:
                raise ValueError(
                    f"Model's output dimension ({last_layer.out_features}) "
                    f"does not match number of classes ({num_classes})"
                )

    print(f"Model is on device: {next(net.parameters()).device}")

    print(
        'Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(args.epoch, args.b, args.lr,
                                                                                            args.gpu, args.seed))

    sysstr = platform.system()
    if (sysstr == "Windows"):
        num_workers = 0
    else:
        num_workers = 8  # 在windows上的进程是0， 在Linux的是8？ 在Windows 在多进程的数据加载时可能会遇到问题？？？？

    pathway = args.data_path  # 默认Linux的问题
    if sysstr == 'Linux':
        pathway = args.data_path

    train_loader, weight_train, number_train = get_weighted_mydataloader(pathway, data_id=0, batch_size=args.b,
                                                                         num_workers=num_workers, shuffle=True)
    
    valid_loader = get_mydataloader(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)

    if args.weight_d > 0:
        reg_loss = Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")

    # 计算类别权重
    classes = np.unique(number_train)
    weights = compute_class_weight('balanced', classes=classes, y=number_train)
    class_weights = torch.FloatTensor(weights).to(device)

    # 修改损失函数，添加权重
    loss_function_CE = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 从0.0001增加到0.001
    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # 使用损失值时改为'min'，使用准确率时用'max'
        factor=0.1, 
        patience=5, 
        verbose=True
    )

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):  # 如果没 log 路径 创建log路径
        os.mkdir(settings.LOG_DIR)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):  # 参数路径
        os.makedirs(checkpoint_path)
    checkpoint_path_pth = os.path.join(checkpoint_path, '{net}-{type}.pth')
    # 添加完整模型保存路径
    complete_model_path = os.path.join(checkpoint_path, '{net}-complete.pt')

    best_acc = 0.0
    Train_Loss = []
    Train_Accuracy = []
    Valid_Loss = []
    Valid_Accuracy = []
    f1_s = []
    best_epoch = 1
    best_weights_path = checkpoint_path_pth.format(net=args.net, type='best')
    # 添加完整模型保存路径格式化
    best_complete_model_path = complete_model_path.format(net=args.net)

    # 在训练循环开始前初始化类别相关的字典
    for i in range(num_classes):  # 使用num_classes而不是硬编码的数字
        global_vars['class_accuracies'][i] = []
        global_vars['class_f1_scores'][i] = []

    # validation_loss = 0
    for epoch in range(1, args.epoch + 1):
        net = train(train_loader, net, optimizer, epoch, loss_function=loss_function_CE, samples_per_cls=number_train)
        acc, validation_loss, fs_valid = eval_training(valid_loader, net, loss_function_CE, epoch)
        
        train_scheduler.step(validation_loss)
        
        # 记录指标
        Train_Loss.append(validation_loss)
        Valid_Loss.append(validation_loss)
        Valid_Accuracy.append(acc.cpu().numpy() if torch.is_tensor(acc) else acc)  # 确保acc在CPU上
        f1_s.append(fs_valid)

        # start to save best performance model
        if epoch > settings.MILESTONES[0] and best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            # 保存模型参数
            torch.save(net.state_dict(), best_weights_path)
            # 保存完整模型（结构+参数）
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validation_loss,
                'accuracy': acc,
                'model': net,  # 保存完整模型
                'args': args,  # 保存模型配置
            }, best_complete_model_path)
    print('best epoch is {}'.format(best_epoch))

    # plot accuracy varying over time
    font_1 = {'weight': 'normal', 'size': 20}
    fig1 = plt.figure(figsize=(12, 9))
    plt.title('Accuracy', font_1)
    index_train = list(range(1, len(Train_Accuracy) + 1))
    
    # 确保数据在绘图前转换为numpy数组
    train_acc_np = np.array(Train_Accuracy)
    valid_acc_np = np.array(Valid_Accuracy)
    
    plt.plot(index_train, train_acc_np, color='skyblue', label='train_accuracy')
    plt.plot(index_train, valid_acc_np, color='red', label='valid_accuracy')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0, 100)
    plt.xlabel('n_iter', font_1)
    plt.ylabel('Accuracy', font_1)

    acc_figuresavedpath = os.path.join(checkpoint_path, 'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath)
    plt.close()  # 添加close防止内存泄漏

    # plot loss varying over time
    fig2 = plt.figure(figsize=(12, 9))
    plt.title('Loss', font_1)
    index_valid = list(range(1, len(Valid_Loss) + 1))
    
    # 确保数据在绘图前转换为numpy数组
    train_loss_np = np.array(Train_Loss)
    valid_loss_np = np.array(Valid_Loss)
    
    plt.plot(index_valid, train_loss_np, color='skyblue', label='train_loss')
    plt.plot(index_valid, valid_loss_np, color='red', label='valid_loss')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0, 100)
    plt.xlabel('n_iter', font_1)
    plt.ylabel('Loss', font_1)

    loss_figuresavedpath = os.path.join(checkpoint_path, 'Loss_curve.png')
    plt.savefig(loss_figuresavedpath)
    plt.close()  # 添加close防止内存泄漏

    # plot f1 score varying over time
    fig3 = plt.figure(figsize=(12, 9))
    plt.title('F1-score', font_1)
    index_fs = list(range(1, len(f1_s) + 1))
    plt.plot(index_fs, f1_s, color='skyblue')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0, 100)
    plt.xlabel('n_iter', font_1)
    plt.ylabel('Loss', font_1)

    fs_figuresavedpath = os.path.join(checkpoint_path, 'F1-score.png')
    plt.savefig(fs_figuresavedpath)
    # plt.show()

    out_txtsavedpath = os.path.join(checkpoint_path, 'output.txt')
    f = open(out_txtsavedpath, 'w+')

    print(
        'Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Data path: {}, Saved path: {}'.format(
            args.seed, args.epoch, args.b, args.lr, args.weight_d, args.gpu, args.data_path, args.save_path),
        file=f)

    print('index: {}; maximum value of validation accuracy: {}.'.format(Valid_Accuracy.index(max(Valid_Accuracy)) + 1,
                                                                        max(Valid_Accuracy)), file=f)
    print('index: {}; maximum value of validation f1-score: {}.'.format(f1_s.index(max(f1_s)) + 1, max(f1_s)), file=f)
    print('--------------------------------------------------')
    print('Validation accuracy: {}'.format(Valid_Accuracy), file=f)
    print('Validation F1-score: {}'.format(f1_s), file=f)

    ######load the best trained model and test testing data  ，测试函数，推理
    best_net = get_network(args)
    best_net.load_state_dict(torch.load(best_weights_path)['model_state_dict'])

    total_num_paras, trainable_num_paras = get_parameter_number(best_net)
    print('The total number of network parameters = {}'.format(total_num_paras), file=f)
    print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)

    best_net.eval()
    number = 0
    correct_test = 0.0
    test_target = []
    test_predict = []

    with torch.no_grad():

        start = time.time()

        for n_iter, (image, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                labels = labels.cuda()

            output = best_net(image)
            output = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            # _, preds = output.topk(5, 1, largest=True, sorted=True)
            # _, preds = output.max(1)
            correct_test += preds.eq(labels).sum()

            if args.gpu:
                labels = labels.cpu()
                preds = preds.cpu()

            test_target.extend(labels.numpy().tolist())
            test_predict.extend(preds.numpy().tolist())

            number += 1

        print('Label values: {}'.format(test_target), file=f)
        print('Predicted values: {}'.format(test_predict), file=f)

        finish = time.time()
        accuracy_test = correct_test.float() / len(test_loader.dataset)
        print('Testing network......', file=f)
        print('Test set: Accuracy: {:.5f}, Time consumed: {:.5f}s'.format(
            accuracy_test,
            finish - start
        ), file=f)

        # Obtain f1_score of the prediction
        fs_test = f1_score(test_target, test_predict, average='macro')
        print('f1 score = {:.5f}'.format(fs_test), file=f)

        kappa_value = cohen_kappa_score(test_target, test_predict)
        print("kappa value = {:.5f}".format(kappa_value), file=f)

        precision_test = precision_score(test_target, test_predict, average='macro')
        print('precision = {:.5f}'.format(precision_test), file=f)

        recall_test = recall_score(test_target, test_predict, average='macro')
        print('recall = {:.5f}'.format(recall_test), file=f)

        # Output the classification report
        print('------------', file=f)
        print('Classification Report', file=f)
        print(classification_report(test_target, test_predict), file=f)

        if not os.path.exists('./results.csv'):
            with open("./results.csv", 'w+') as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(['index', 'accuracy', 'f1-score', 'precision', 'recall', 'kappa', 'time_consumed'])

        with open("./results.csv", 'a+') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow(
                [args.seed, accuracy_test, fs_test, precision_test, recall_test, kappa_value, finish - start])

        # Show the confusion matrix so that it can help us observe the results more intuitively
        def show_confusion_matrix(validations, predictions):
            matrix = confusion_matrix(validations, predictions)  # No one-hot
            # matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
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
            cm_figuresavedpath = os.path.join(checkpoint_path, 'Confusion_matrix.png')
            plt.savefig(cm_figuresavedpath)


        show_confusion_matrix(test_target, test_predict)

    if args.gpu:
        print('GPU INFO.....', file=f)
        print(torch.cuda.memory_summary(), end='', file=f)

    # 在训练结束后添加绘图函数
    def plot_training_curves(checkpoint_path, learning_rates, class_accuracies, class_f1_scores, Class_labels):
        # 1. 学习率变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.title('Learning Rate over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(checkpoint_path, 'learning_rate_curve.png'))
        plt.close()

        # 2. 每个类别的准确率变化
        plt.figure(figsize=(12, 8))
        for i, label in enumerate(Class_labels):
            plt.plot(class_accuracies[i], label=label)
        plt.title('Class-wise Accuracy over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_path, 'class_wise_accuracy.png'))
        plt.close()

        # 3. 每个类别的F1分数变化
        plt.figure(figsize=(12, 8))
        for i, label in enumerate(Class_labels):
            plt.plot(class_f1_scores[i], label=label)
        plt.title('Class-wise F1 Score over Training')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_path, 'class_wise_f1.png'))
        plt.close()

    def plot_roc_curve(y_test, y_score, n_classes, Class_labels, checkpoint_path):
        # 将标签转换为one-hot编码
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # 计算每个类别的ROC曲线和ROC面积
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        
        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{Class_labels[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_path, 'roc_curve.png'))
        plt.close()

    # 在测试部分添加ROC曲线绘制
    with torch.no_grad():
        # 获取模型输出的概率分布
        all_probs = []
        for n_iter, (image, labels) in enumerate(test_loader):
            if args.gpu:
                image = image.cuda()
            output = best_net(image)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
        
        # 绘制ROC曲线
        plot_roc_curve(test_target, all_probs, len(Class_labels), Class_labels, checkpoint_path)

    # 在训练结束后调用绘图函数
    plot_training_curves(checkpoint_path, 
                        global_vars['learning_rates'], 
                        global_vars['class_accuracies'], 
                        global_vars['class_f1_scores'], 
                        Class_labels)

    # 动态修改输出层
    if hasattr(net, 'head'):
        in_features = net.head.in_features
        net.head = nn.Linear(in_features, num_classes).to(device)
    else:
        print("Warning: Could not find output layer 'head' to modify")
