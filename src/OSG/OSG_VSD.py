import torch
import numpy as np

def my_Tloss(input_prediction_stack,input_groundtruth_stack, device=torch.device("cuda")):
    total_loss = 0
    input_prediction_list = input_prediction_stack.unbind()
    input_groundtruth_list = input_groundtruth_stack.unbind()
    for the_ind in range(len(input_prediction_list)):
        input_prediction, input_groundtruth = input_prediction_list[the_ind], input_groundtruth_list[the_ind]
        T_pred = torch.masked_select(input_prediction, ~torch.isnan(input_prediction))
        t = torch.masked_select(input_groundtruth, ~torch.isnan(input_groundtruth))
        T_gt = torch.zeros(int(t[-1]), device=device)
        for scene_end in t[:-1]:
            T_gt[int(scene_end)] = 1
        total_loss = total_loss + torch.masked_select(T_pred[:-1], T_gt.ge(1)).clamp(min=1e-3).log().neg().sum()
    return total_loss

class DIST(torch.nn.Module):
    def __init__(self, feature_sizes, BN=False, DO=0.0, dist_type='EMBEDDING', dist_metric='cosine', device=torch.device("cuda")):
        super(DIST, self).__init__()
        self.device = device
        self.feature_sizes =feature_sizes
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(DO)
        self.network = torch.nn.Sequential()
        for layer_num in range(0, len(feature_sizes)-1):
            self.network.add_module('FC'+str(layer_num),torch.nn.Linear(feature_sizes[layer_num], feature_sizes[layer_num+1]))
            if isinstance(BN, list):
                if BN[layer_num]:
                    self.network.add_module('BN'+str(layer_num),torch.nn.BatchNorm1d(feature_sizes[layer_num+1]))
            else:
                if BN:
                    self.network.add_module('BN'+str(layer_num),torch.nn.BatchNorm1d(feature_sizes[layer_num+1]))
            if layer_num < len(feature_sizes)-2:
                self.network.add_module('ACT'+str(layer_num),self.activation)
                if DO > 0.0:
                    self.network.add_module('DO' + str(layer_num),self.dropout)
        self.dist_type = dist_type
        self.dist_metric = dist_metric

    def forward(self, input_x):
        if list(input_x.shape)[0] > 1:
            print('Warning - expected batch size 1')
        x = input_x.squeeze(0)

        if self.dist_type == 'DIST':
            x_new = torch.cat((x.repeat(x.shape[0],1),x.repeat(1,x.shape[0]).view(x.shape[0]*x.shape[0],-1)),1)
        elif self.dist_type == 'EMBEDDING':
            x_new = x
        else:
            print('Warning - unrecognized dist_type. Performing EMBEDDING.')
            x_new = x

        x_new = self.network(x_new)

        if self.dist_type=='EMBEDDING':
            if self.dist_metric=='cosine':
                x_new_corr = x_new.matmul(x_new.t())
                x_new_square = torch.masked_select(x_new_corr, torch.eye(x_new.shape[0], device=self.device).ge(1))
                x_new_square_rows = x_new_square[:, None].repeat(1, x_new.shape[0])
                x_new_square_cols = x_new_square.t().repeat(x_new.shape[0], 1)
                D = (1.0 - x_new_corr / (x_new_square_rows * x_new_square_cols).clamp(min=1e-8).sqrt()) / 2.0
            elif self.dist_metric=='euclidean':
                D = torch.norm(x_new[:, None] - x_new, dim=2, p=2)
            else:
                print('Warning - unrecognized dist_metric. Performing euclidean.')
                D = torch.norm(x_new[:, None] - x_new, dim=2, p=2)
        elif self.dist_type=='DIST':
            D = x_new.view(x.shape[0],-1)

        D.unsqueeze_(0)
        return D

class D_SUM_CALC(torch.nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super(D_SUM_CALC, self).__init__()
        self.device = device
    def forward(self, input_D):
        if list(input_D.shape)[0] > 1:
            print('Warning - expected batch size 1')
        D = input_D.squeeze(0)
        N = list(D.shape)[0]
        D_sum = torch.zeros(N,N, device=self.device)
        # diagonal
        for ii in range(N):
            D_sum[ii,ii] = D[ii,ii]
        # second diagonal
        for ii in range(0, N-1):
            D_sum[ii, ii+1] = D[ii:ii+1+1, ii:ii+1+1].sum()
            D_sum[ii+1, ii] = D[ii, ii+1]
        # rest
        for oo in range(2, N):
            for ii in range(0, N - oo):
                D_sum[ii, ii + oo] = D[ii, ii + oo] + D[ii + oo, ii] + D_sum[ii, ii + oo - 1] + D_sum[ii + 1, ii + oo] - D_sum[ii + 1, ii + oo - 1]
                D_sum[ii + oo, ii] = D_sum[ii, ii + oo]
        D_sum.unsqueeze_(0)
        return D_sum

class C_TABLE_ALL(torch.nn.Module):
    def __init__(self, K, device=torch.device("cuda")):
        super(C_TABLE_ALL, self).__init__()
        self.K = K
        self.device = device
    def forward(self, input_D_sum):
        if list(input_D_sum.shape)[0] > 1:
            print('Warning - expected batch size 1')
        D_sum = input_D_sum.squeeze(0)
        N = list(D_sum.shape)[0]
        K = self.K
        C = torch.zeros(N, K, device=self.device)
        C_all = -1 * torch.ones(N, K, N, device=self.device)
        the_softmin = torch.nn.Softmin(dim=0)
        for nn in range(N):
            C[nn, 0] = D_sum[nn, N-1]
            C_all[nn, 0, N-1] = 1.0
        for kk in range(1, K):
            for nn in range(0, N - kk):
                temp = torch.empty(N - kk - nn, device=self.device)
                for ii in range(nn, N - kk):
                    temp[ii-nn] = D_sum[nn, ii] + C[ii + 1, kk - 1]
                C_all[nn, kk, nn:N-kk] = the_softmin(temp)
                C[nn, kk] = torch.min(temp)
        C.unsqueeze_(0)
        C_all.unsqueeze_(0)
        return C, C_all

class OSG_C(torch.nn.Module):
    def __init__(self, feature_sizes, K_max=30, BN=False, DO=0.0, dist_type='EMBEDDING', dist_metric='cosine', device=torch.device("cuda")):
        super(OSG_C, self).__init__()
        self.feature_sizes = feature_sizes
        self.K_max = K_max
        self.DIST_FUNC = DIST(feature_sizes,BN,DO,dist_type,dist_metric, device)
        self.D_SUM_CALC = D_SUM_CALC(device)
        self.C_TABLE_ALL = C_TABLE_ALL(K_max, device)
        self.device = device
    def forward(self, x):
        T_list = list()
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        for x_input in x.unbind():
            x_input = torch.masked_select(x_input, ~torch.isnan(x_input)).view(1, -1, x_input.shape[1])
            D = self.DIST_FUNC(x_input)
            D_sum = self.D_SUM_CALC(D)
            __, C_all = self.C_TABLE_ALL(D_sum)
            T_pred_all = torch.zeros(C_all.shape[3], device=self.device)
            for ind in range(C_all.shape[3]):
                T_pred_all[ind] = torch.masked_select(C_all[0, :, :, ind], C_all[0, :, :, ind].ge(0)).mean()
            the_padding = torch.nn.modules.padding.ConstantPad1d((0, x.shape[1] - T_pred_all.shape[0]), float('nan'))
            T_list.append(the_padding(T_pred_all))
        T_out = torch.stack(T_list)
        return T_out
