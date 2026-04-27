import scipy.io as sio
import numpy as np
import os
import cv2
import torch
from torchvision import transforms
import metrics
import mat73
from Model import Fusion

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def im2uint8(x):
    x = (x) * 255
    out = np.clip(x, 0, 255) + 0.5
    return out.astype(np.uint8)

if __name__ == '__main__':
    net = Fusion().cuda()
    net.load_state_dict(torch.load("./model/state_dicr_200.pkl", map_location=torch.device('cuda')))
    net.eval()

    test_path = 'F:\\unregister_Hyperspectral_Image_Fusion_Benchmarkx4\\houton\\data\\test'
    psnr, rmse, ergas, sam, uiqi, ssim = np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)

    for i in range(8):
        ind = i + 1
        path = str(i + 1) + '.mat'
        print('processing for %d' % ind)

        # Load hyperspectral data
        source_hs_path = os.path.join(test_path, 'hs', path)
        data = mat73.loadmat(source_hs_path)
        data = torch.FloatTensor(data['I']).permute(2, 0, 1).unsqueeze(0).cuda()

        # Load multispectral data
        source_ms_path = os.path.join(test_path, 'ms', path)
        data1 = mat73.loadmat(source_ms_path)
        data1 = torch.FloatTensor(data1['I']).permute(2, 0, 1).unsqueeze(0).cuda()

        source_mi_path  = os.path.join(test_path, 'MI_hs', path)
        MI = mat73.loadmat(source_mi_path)
        MI = torch.FloatTensor(MI['MIsub']).permute(2, 0, 1).unsqueeze(0).cuda()

        # Load ground truth data

        source_ms_path = os.path.join(test_path, 'gt', path)
        GT = mat73.loadmat(source_ms_path)
        GT = torch.FloatTensor(GT['I']).permute(2, 0, 1).unsqueeze(0).cuda()

        # Convert GT to uint8 for metrics

        with torch.no_grad():
            data_get1 = net(data, data1,MI)

        data_get = data_get1[0]
        #reg = data_get1[1]
        X_reg = data_get1[1]


        # Convert tensors to numpy arrays for metrics
        X_reg = X_reg.cpu().detach().numpy()
        X_reg = np.transpose(X_reg, [0, 2, 3, 1])
        X_reg = np.reshape(X_reg, (64, 64, 50))
        X_reg = im2uint8(X_reg)

        #reg = reg.cpu().detach().numpy()
        #reg = np.transpose(reg, [0, 2, 3, 1])
        #reg = np.reshape(reg, (256, 256, 31))
        #reg = im2uint8(reg)

        # Similar conversion for Z2, Z3, Z4, Z5

        # Use metrics.metric function to calculate evaluation metrics
        psnr[i], rmse[i], ergas[i], sam[i], uiqi[i], ssim[i], data_get = metrics.metric(data_get, GT)

        # Save intermediate results
        sio.savemat('./get/eval_%d.mat' % ind, {'b': data_get})
        sio.savemat('./get/evalX_reg_%d.mat' % ind, {'b': X_reg})
        #sio.savemat('./get/evalreg_%d.mat' % ind, {'b': reg})

        torch.cuda.empty_cache()
    print('PSNR is: %.4f, RMSE is: %.4f, ERGAS is: %.4f, SAM is: %.4f, UIQI is: %.4f, SSIM is: %.4f' %
          (np.mean(psnr), np.mean(rmse), np.mean(ergas), np.mean(sam), np.mean(uiqi), np.mean(ssim)))
