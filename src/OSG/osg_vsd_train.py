import torch
import OSG_VSD as OSG
import numpy as np
np.set_printoptions(linewidth=300)
import osg_vsd_dataset
import OptimalSequentialGrouping


def CLossTest(data_folder_path='h5/', modality='visual', num_iters=101, stop_param=0.75):

    if modality=='visual':
        path_to_h5 = data_folder_path+'h5_visual/'
        d, K_max = 2048, 5
        BN = True
        DO = 0.0
        dist_metric = 'cosine'
        dist_type = 'EMBEDDING'
        feature_sizes = [d, 3000, 3000, 1000, 100]
        learning_rate = 0.005
        weight_decay = 0  # 1e-2
    elif modality=='audio':
        path_to_h5 = data_folder_path+'h5_audio/'
        d, K_max = 128, 5
        BN = True
        DO = 0.0
        dist_metric = 'cosine'
        dist_type = 'EMBEDDING'
        feature_sizes = [d, 200, 200, 100, 20]
        learning_rate = 0.005
        weight_decay = 0  # 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    vsd_dataset = osg_vsd_dataset.OSG_VSD_DATASET(path_to_h5=path_to_h5,device=device)

    vsd_dataloader = torch.utils.data.DataLoader(vsd_dataset, collate_fn=osg_vsd_dataset.my_collate)

    OSG_model = OSG.OSG_C(feature_sizes, K_max=K_max, BN=BN, DO=DO, dist_type=dist_type, dist_metric=dist_metric, device=device)

    OSG_model.to(device)

    optimizer = torch.optim.Adam(OSG_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print('Starting')
    print('network',feature_sizes,'dist_type',dist_type,'dist_metric',dist_metric,'stop_param',stop_param,'modality',modality)

    first_loss = 0
    for iteration in range(num_iters):

        optimizer.zero_grad()
        all_loss = 0

        for a_batch in vsd_dataloader:
            x, t = a_batch

            T_pred = OSG_model(x.to(device))

            loss = OSG.my_Tloss(T_pred.to(device), t.to(device), device=device)
            all_loss += loss.item()

            loss.backward()

        if iteration == 0:
            first_loss = all_loss

        optimizer.step()

        OSG_np = OptimalSequentialGrouping.OptimalSequentialGrouping()

        F_trn = 0
        for an_index in range(len(vsd_dataset)):
            x_orig, t_orig = vsd_dataset[an_index]
            t = t_orig.cpu().numpy()
            D_temp = OSG_model.module.DIST_FUNC(x_orig.unsqueeze(0))
            D_new = D_temp.squeeze(0).cpu().detach().numpy()
            boundaries_new = OSG_np.blockDivideDSum(D_new, t.size)
            F_temp, __, __ = OSG_np.FCO(boundaries_new, t)
            F_trn += F_temp

        print('Iteration '+str(iteration)+ ', loss: '+str(all_loss)+', F-score: '+str(F_trn))

        if all_loss < stop_param*first_loss:
            break

    print('finished')

if __name__ == "__main__":
    CLossTest(num_iters=5)