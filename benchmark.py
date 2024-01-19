import os
import sys
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
from tool import dataSplit
from tool import getData
from timm.models import vision_transformer
from tool.lr_sched import adjust_learning_rate_milestones, adjust_learning_rate
import matplotlib.pyplot as plt


def main(train_images_path, train_images_label, val_images_path, val_images_label, dataset_name):
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    # np.random.seed(42)
    model_name = dir_name = '224Abenchmark1'
    warmup_epochs = 10
    yhq = 'ms'
    opt = 'AdamW'
    # model = vision_transformer.vit_small_patch16_224_in21k(num_classes=5, pretrained=True, img_size=896)
    model = vision_transformer.vit_small_patch16_224_in21k(num_classes=5, pretrained=True)
    if opt == 'SGD':
        label_smoothing = 0.05
        epochs = 120
        gamma = 0.8
        momentum = 0.9
        lr = 1e-3
        milestones = [20, 35, 60, 80]
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if opt == 'Adam':
        label_smoothing = 0.01
        epochs = 120
        lr = 1e-4
        gamma = 0.5
        milestones = [20, 35, 60, 80]
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if opt == 'AdamW':
        label_smoothing = 0.1
        epochs = 110
        lr = 1e-4
        gamma = 0.5
        milestones = [20, 35, 60]
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_dataset = getData.getTrainDataSet(train_images_path, train_images_label, dataset_name)
    train_num = train_dataset.__len__()
    samper = getData.balanceData(train_images_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=samper, batch_size=batch_size, num_workers=nw)

    validate_dataset = getData.getValDataSet(val_images_path, val_images_label, dataset_name)
    val_num = validate_dataset.__len__()
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    os.makedirs(dir_name, exist_ok=True)
    model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = 0.0
    best_epoch = 0
    train_steps = len(train_loader)
    dev_steps = len(validate_loader)
    saveTrainLoss = []
    saveTrainAcc = []
    saveDevLoss = []
    saveValidAcc = []
    saveValidSen = []
    saveAUC = []
    savekappa = []
    save_lr = []
    for epoch in range(1, epochs+1):
        # train
        model.train()
        if yhq == 'ms':
            adjust_learning_rate_milestones(optimizer, epoch, warmup_epochs=warmup_epochs,
                                            milestones=milestones, gamma=gamma, max_lr=lr)
        if yhq == 'cos':
            adjust_learning_rate(optimizer, epoch, epochs, warmup_epochs, lr)
        save_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        running_loss = 0.0
        dev_loss = 0.0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch, epochs, loss)
            train_total += images.shape[0]
            predict = torch.max(outputs, dim=1)[1]
            train_correct += torch.eq(predict, labels.to(device)).sum().item()
        saveTrainAcc.append(100 * train_correct / train_total)

        # validate
        model.eval()
        wholePre = []
        wholeLabel = []
        wholeBeforeBinaryzation = []
        correct = 0
        total = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                total += val_images.shape[0]
                outputs = model(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                dev_loss += loss.item()

                predict = torch.max(outputs, dim=1)[1]
                correct += torch.eq(predict, val_labels.to(device)).sum().item()
                wholePre = np.append(wholePre, predict.cpu().detach().numpy())
                outputs = torch.softmax(outputs, 1)
                if wholeBeforeBinaryzation.__len__() == 0:
                    wholeBeforeBinaryzation = outputs.cpu().detach().numpy()
                else:
                    wholeBeforeBinaryzation = np.vstack((wholeBeforeBinaryzation, outputs.cpu().detach().numpy()))

                val_labels = val_labels.detach().numpy()
                wholeLabel = np.append(wholeLabel, val_labels)

        # scheduler.step()
        acc = 100 * correct / total
        sen = 100 * metrics.recall_score(wholeLabel, wholePre, average='macro')
        auc = 100 * metrics.roc_auc_score(wholeLabel, wholeBeforeBinaryzation, multi_class='ovr')
        kappa = metrics.cohen_kappa_score(wholeLabel, wholePre, weights='quadratic')
        print('[epoch %d] train_loss: %.3f dev_loss: %.3f'
              % (epoch, running_loss / train_steps, dev_loss / dev_steps))
        print('val_accuracy: %.3f Sen: %.3f Auc: %.3f Kappa: %.3f'
              % (acc, sen, auc, kappa))
        saveTrainLoss = np.append(saveTrainLoss, running_loss / train_steps)
        saveDevLoss = np.append(saveDevLoss, dev_loss / dev_steps)
        saveAUC = np.append(saveAUC, auc)
        saveValidAcc = np.append(saveValidAcc, acc)
        saveValidSen = np.append(saveValidSen, sen)
        savekappa = np.append(savekappa, kappa)
        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            torch.save(model.state_dict(), dir_name + '/' + model_name + '.pth')

    savedata = np.vstack((saveTrainLoss, saveDevLoss, saveValidAcc,
                          saveAUC, saveValidSen, savekappa, save_lr, saveTrainAcc))
    savedata = pd.DataFrame(savedata)
    savedata.rename(index={0: 'TrainLoss', 1: 'DevLoss', 2: 'DevAcc',
                           3: 'AUC', 4: 'Sen', 5: 'Kappa', 6: 'LR', 7: 'TrainAcc'}, inplace=True)
    writer = pd.ExcelWriter(dir_name + '/' + model_name + '.xlsx')
    savedata.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    save_info = []
    save_info.extend([model_name, 'batch_size:', batch_size, opt, lr, dataset_name, label_smoothing, best_epoch, best_acc])
    f = open(dir_name + '/' + 'info.txt', 'w')
    for i in save_info:
        f.write(str(i))
        f.write(' ')
    f.close()

    xa = np.arange(epochs)
    figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
    axes[0, 0].plot(xa, saveTrainLoss, label='TrainLoss')
    axes[0, 0].plot(xa, saveDevLoss, label='DevLoss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss')
    axes[0, 1].plot(xa, save_lr)
    axes[0, 1].set_title('LR')
    axes[1, 0].plot(xa, saveTrainAcc, label='TrainAcc')
    axes[1, 0].plot(xa, saveValidAcc, label='ValAcc')
    axes[1, 0].legend()
    axes[1, 0].set_title('ACC')
    axes[1, 1].plot(xa, saveAUC)
    axes[1, 1].set_title('AUC')
    axes[2, 0].plot(xa, savekappa)
    axes[2, 0].set_title('Kappa')
    axes[2, 1].plot(xa, saveValidSen)
    axes[2, 1].set_title('Sen')
    plt.savefig(dir_name + '/fig.png')


    print('Best Acc: {:.5f}'.format(np.max(saveValidAcc)))
    print('Best epoch:', end=' ')
    print(best_epoch)
    print('Best AUC: {:.5f}'.format(np.max(saveAUC)))
    print('Best kappa: {:.5f}'.format(np.max(savekappa)))
    print('Finished Training')
    return model, dir_name


def model_test(train_images_path, train_images_label, model, dir_name, dataset_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("using {} device.".format(device))

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_dataset = getData.getValDataSet(train_images_path, train_images_label, dataset_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    model.load_state_dict(torch.load(dir_name + '/' + dir_name + '.pth', map_location='cpu'))
    model.to(device)

    os.makedirs(dir_name + '/result', exist_ok=True)

    loss_function = nn.CrossEntropyLoss()

    test_steps = len(test_loader)
    saveTestLoss = []
    saveTestAcc = []
    saveTestAUC = []
    saveTestkappa1 = []
    saveTestkappa2 = []
    saveTestSen = []

    # validate
    model.eval()
    wholePre = []
    wholeLabel = []
    wholeBeforeBinaryzation = []
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            total += val_images.shape[0]
            outputs = model(val_images.to(device))
            loss = loss_function(outputs, val_labels.to(device))
            test_loss += loss.item()

            predict = torch.max(outputs, dim=1)[1]
            correct += torch.eq(predict, val_labels.to(device)).sum().item()
            wholePre = np.append(wholePre, predict.cpu().detach().numpy())
            outputs = torch.softmax(outputs, 1)
            if wholeBeforeBinaryzation.__len__() == 0:
                wholeBeforeBinaryzation = outputs.cpu().detach().numpy()
            else:
                wholeBeforeBinaryzation = np.vstack((wholeBeforeBinaryzation, outputs.cpu().detach().numpy()))

            val_labels = val_labels.detach().numpy()
            wholeLabel = np.append(wholeLabel, val_labels)


    acc = 100 * correct / total
    auc = 100 * metrics.roc_auc_score(wholeLabel, wholeBeforeBinaryzation, multi_class='ovo')
    kappa1 = metrics.cohen_kappa_score(wholeLabel, wholePre, weights='quadratic')
    kappa2 = metrics.cohen_kappa_score(wholeLabel, wholePre, weights='linear')
    sen = 100 * metrics.recall_score(wholeLabel, wholePre, average='macro')
    c = metrics.confusion_matrix(wholeLabel, wholePre)
    dispcm = metrics.ConfusionMatrixDisplay(confusion_matrix=c, display_labels=['0', '1', '2', '3', '4'])
    dispcm.plot()
    plt.savefig(dir_name + '/result/confusion_matrix' + '.jpg')
    plt.show()
    print('test_loss: %.3f accuracy: %.3f AUC: %.3f Kappa1: %.3f Kappa2: %.3f'
            % (test_loss / test_steps, acc, auc, kappa1, kappa2))
    saveTestLoss = np.append(saveTestLoss, test_loss / test_steps)
    saveTestAUC = np.append(saveTestAUC, auc)
    saveTestAcc = np.append(saveTestAcc, acc)
    saveTestkappa1 = np.append(saveTestkappa1, kappa1)
    saveTestkappa2 = np.append(saveTestkappa2, kappa2)
    saveTestSen = np.append(saveTestSen, sen)

    savedata = np.vstack((saveTestLoss, saveTestAcc, saveTestAUC, saveTestkappa1, saveTestkappa2, saveTestSen))
    savedata = pd.DataFrame(savedata)
    savedata.rename(index={0: 'TestLoss', 1: 'ACC', 2: 'AUC', 3: 'Kappa1', 4: 'Kappa2', 5: 'Sen'}, inplace=True)
    writer = pd.ExcelWriter(dir_name + '/result/result' + '.xlsx')
    savedata.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    print('Finished')

if __name__ == '__main__':
    dataset_name = 'APTOS'
    train_images_path, train_images_label, val_images_path, val_images_label = dataSplit.read_split_data(
        '../data/' + dataset_name + '/train')
    model, dir_name = main(train_images_path, train_images_label, val_images_path, val_images_label, dataset_name)
    train_images_path, train_images_label, val_images_path, val_images_label = dataSplit.read_split_data(
        '../data/' + dataset_name + '/test', val_rate=0)
    model_test(train_images_path, train_images_label, model, dir_name, dataset_name)
