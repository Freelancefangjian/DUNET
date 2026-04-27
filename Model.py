import torch
import torch.nn as nn
import kornia.utils as KU
import kornia.filters as KF
import torch.nn.functional as F
import cope
device = torch.device("cuda:0")
class Fusion(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2):
        super(Fusion, self).__init__()
        self.DM = cope.DenseMatcher()
        self.ST = cope.SpatialTransformer(16, 16, True)
        self.FN = cope.FusionNet()
        self.alpha_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.beta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.gamma_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))

        self.alpha_1 = torch.nn.Parameter(torch.tensor(0.1))
        self.beta_1 = torch.nn.Parameter(torch.tensor(0.1))
        self.gamma_1 = torch.nn.Parameter(torch.tensor(0.1))
        self.delta_1 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_1 = torch.nn.Parameter(torch.tensor(0.9))

        self.alpha_2 = torch.nn.Parameter(torch.tensor(0.1))
        self.beta_2 = torch.nn.Parameter(torch.tensor(0.1))
        self.gamma_2 = torch.nn.Parameter(torch.tensor(0.1))
        self.delta_2 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_2 = torch.nn.Parameter(torch.tensor(0.9))

        self.alpha_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.beta_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.gamma_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.delta_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_3 = torch.nn.Parameter(torch.tensor(0.9))

        self.alpha_4 = torch.nn.Parameter(torch.tensor(0.1))
        self.beta_4 = torch.nn.Parameter(torch.tensor(0.1))
        self.gamma_4 = torch.nn.Parameter(torch.tensor(0.1))
        self.delta_4 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_4 = torch.nn.Parameter(torch.tensor(0.9))

        self.n = 5
        self.conv_upsample = torch.nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=4,stride=4, padding=0)
        self.conv_downsample = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3,stride=4, padding=3 // 2)
        self.conv_downsample_3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=4,padding=3 // 2)
        self.Conv31_31 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, padding=(1, 1))
        self.Conv3_31 = nn.Conv2d(in_channels=4, out_channels=50, kernel_size=3, padding=(1, 1))
        self.Conv31_3 = nn.Conv2d(in_channels=50, out_channels=4, kernel_size=3, padding=(1, 1))
        self.prox = cope.Prox()
        self.Upsa = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=False)
    def recon(self, Z, X_reg,V,Y, ir_reg1,D1,D2,D3, id_layer):
        if id_layer == 0 :
            ALPHA = self.alpha_0
            BETA = self.beta_0
            GAMMA = self.gamma_0
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1 :
            ALPHA = self.alpha_1
            BETA = self.beta_1
            GAMMA = self.gamma_1
            DELTA = self.delta_1
            ETA = self.eta_1
        elif id_layer == 2:
            ALPHA = self.alpha_2
            BETA = self.beta_2
            GAMMA = self.gamma_2
            DELTA = self.delta_2
            ETA = self.eta_2
        elif id_layer == 3:
            ALPHA = self.alpha_3
            BETA = self.beta_3
            GAMMA = self.gamma_3
            DELTA = self.delta_3
            ETA = self.eta_3
        elif id_layer == 4:
            ALPHA = self.alpha_4
            BETA = self.beta_4
            GAMMA = self.gamma_4
            DELTA = self.delta_4
            ETA = self.eta_4

        Z = Z - ETA * (self.conv_upsample(X_reg - self.conv_downsample(Z)) + ALPHA * self.Conv3_31(
            (Y - self.Conv31_3(Z))) + DELTA * (Z - V-D3))
        X_reg = X_reg - ETA * ((X_reg - self.conv_downsample(Z)) + BETA * self.Conv3_31(
            self.Conv31_3(X_reg) - self.conv_downsample_3(Y)-D1) + GAMMA * (X_reg - ir_reg1-D2))
        D3 = D3-(Z - V)
        D1 = D1-(self.Conv31_3(X_reg) - self.conv_downsample_3(Y))
        D2 = D2 - (X_reg - ir_reg1)
        ################################################################
        return Z,X_reg,D1,D2,D3
    def forward(self, src, tgt, mi, type='ir2vis'):
        #disp = self.DM(src, tgt)['ir2vis']
        ir_reg = src
        X_reg = src
        Y = tgt
        Z = self.conv_upsample(src)
        #V#V = Z
        p,c,h,w = src.shape
        p1, c1, h1, w1 = tgt.shape
        D1 = torch.zeros(p,c1,h,w).cuda(device)
        D2 = torch.zeros(p,c,h,w).cuda(device)
        D3 = torch.zeros(p, c, h1, w1).cuda(device)
        V = torch.zeros(p, c, h1, w1).cuda(device)
        ir_reg = mi
        #for i in range(self.n):

        Z, X_reg,D1,D2,D3 = self.recon(Z, X_reg, V, Y, ir_reg,D1,D2,D3, id_layer=0)
        V= self.prox(Z-D3)

        Z, X_reg,D1,D2,D3 = self.recon(Z, X_reg, V, Y, ir_reg,D1,D2,D3, id_layer=1)
        V = self.prox(Z-D3)


        Z, X_reg,D1,D2,D3 = self.recon(Z, X_reg, V, Y, ir_reg,D1,D2,D3, id_layer=2)
        V = self.prox(Z-D3)


        Z, X_reg,D1,D2,D3 = self.recon(Z, X_reg, V, Y, ir_reg, D1,D2,D3, id_layer=3)
        V = self.prox(Z-D3)


        #在每个阶段中对配准的图像再进行新的一轮配准
        Z, X_reg,D1,D2,D3 = self.recon(Z, X_reg, V, Y, ir_reg,D1,D2,D3, id_layer=4)
        #V = self.prox(Z)

        return Z, X_reg

