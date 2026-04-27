import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from DataSet import DataSet
from config import FLAGES
import torch.utils.data as Data
def l2_penaalty(w):
    return (w**2).sum()/2
import numpy as np
import time
import Model
import math
def PSNR(img1, img2):
    mse_sum  = (img1  - img2 ).pow(2)
    mse_loss = mse_sum.mean(2).mean(2)
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return mse_loss, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((2, 3, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def forward(self, y_true, y_pred):
        # Normalize the vectors
        y_true_normalized = F.normalize(y_true, p=2, dim=-1)
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)

        # Calculate the dot product
        dot_product = torch.sum(y_true_normalized * y_pred_normalized, dim=-1)

        # Calculate the spectral angle
        angle = torch.acos(torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7))

        # Convert angle to degrees and use it as the loss
        loss = torch.mean(torch.rad2deg(angle))

        return loss

from tqdm import tqdm
import matplotlib.pyplot as plt
def plot_acc_loss(loss, psnr, num):
    x2 = range(0, num)
    plt.subplot(2, 1, 1)
    plt.plot(x2, loss, '-', linewidth = 1, label = "loss", color="orange")
    #plt.title("loss")
    plt.grid()
    #plt.savefig('./loss.jpg')
    plt.subplot(2, 1, 2)
    plt.plot(x2, psnr, '-', linewidth=1, label="PSNR", color="red")
    #plt.title("PSNR")
    plt.grid()
    plt.savefig('./psnr&loss.jpg')
    plt.show()
    plt.pause(0.1)
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1):
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        dice_coefficient = (2. * intersection + smooth) / (union + smooth)

        loss = 1 - dice_coefficient
        return loss
import warnings
import scipy.io as sio
import mat73
class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss
if __name__ == '__main__':
    #freeze_support()
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    #dataset = DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
     #                 FLAGES.stride)
    dataFile = 'F:\\unregister_Hyperspectral_Image_Fusion_Benchmarkx4\\houton\\data\\dataset.mat'
    data = mat73.loadmat(dataFile)
    # HR = np.transpose(dataset.gt, [3, 1, 2])
    MSI = torch.from_numpy(np.transpose(data['MSI11'], [3, 2, 0, 1]))
    #MSI = MSI[1:100,:,:,:]
    # MSI = convert_image_np(MSI)
    HSI =  torch.from_numpy(np.transpose(data['HSI11'], [3, 2, 0, 1]))
    #HSI = HSI[1:100, :, :, :]
    GT = torch.from_numpy(np.transpose(data['GT11'], [3, 2, 0, 1]))
    #GT = GT[1:100, :, :, :]
    MI = torch.from_numpy(np.transpose(data['MI_HSI'], [3, 2, 0, 1]))
    #MI = MI[1:100, :, :, :]
    #GT = GT[1:10, :, :, :]
    torch_dataset = Data.TensorDataset(MSI, HSI, GT, MI)
    loader = Data.DataLoader(dataset = torch_dataset, batch_size=8, shuffle=True, num_workers=2)

    device = torch.device("cuda:0")
    net = Model.Fusion()
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    net.to(device)
    import torch.optim as optim

    sam_loss = SAMLoss()
    criterion = nn.L1Loss().to(device)
    criterion1 = nn.MSELoss()
    WEIGHT_DECAY = 1e-8
    optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 700, 900], 0.2)
    min_loss = 1.0
    #net.load_state_dict(torch.load("./model/state_dicr_90.pkl", map_location=torch.device('cuda')))
    pbar = tqdm(range(210))
    plt.ion()
    loss_list = []
    psnr_list = []
    dice_loss = DiceLoss()
    criterion_fft = fftLoss()
    for epoch in pbar:  # loop over the dataset multiple times

        running_loss = 0.0
        mpsnr = 0.0

        for i, data in enumerate(loader, 0):
            # get the inputs
            MSI1, HSI1,  GT1, MI = data
            MSI1 = MSI1.type(torch.FloatTensor)
            HSI1 = HSI1.type(torch.FloatTensor)
            MI = MI.type(torch.FloatTensor)
            GT1 = GT1.type(torch.FloatTensor)

            MSI1 = MSI1.cuda(device)
            MI = MI.cuda(device)
            HSI1 = HSI1.cuda(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, X_reg = net(HSI1, MSI1, MI)
            #loss = F.nll_loss(output, MSI1)
            #outputs = net(MSI1, HSI1)
            GT1 = GT1.cuda(device)
            GT11 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=False)(GT1)
            loss = criterion(outputs, GT1) +0.5*criterion1(X_reg, GT11) + 0.1 * criterion_fft(outputs, GT1)
            mse, psnr = PSNR(outputs, GT1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            mpsnr += psnr
        scheduler.step()
        loss_list.append(running_loss / (i + 1))
        psnr_list.append(mpsnr / (i + 1))
        plot_acc_loss(loss_list, psnr_list, epoch + 1)
        time.sleep(0.001)
        current_learning_rate = optimizer.param_groups[0]['lr']
        pbar.set_description(
            "Processing %s, loss: %.7f, PSNR:%.3f, rate:%.5f" % (epoch + 1, running_loss / (i + 1), mpsnr / (i + 1),current_learning_rate))
        if epoch % 10 == 0:  # print every 2000 mini-batches
            torch.save(net.state_dict(), './model/state_dicr_{}.pkl'.format(epoch))

    print('Finished Training')