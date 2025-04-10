import numpy as np
import time
import torch
def get_time(pde,data,n_step=100,k=2):
    listTimePDE = []
    listTimeBC  = []
    for it in range(n_step+k):
        # pde.optimizer.zero_grad()

        torch.cuda.synchronize()
        tin_f=time.time()
        loss_f   = pde.loss_PDE(data.X_train)
        torch.cuda.synchronize()
        tout_f=time.time()
        dt_f = tout_f-tin_f
        listTimePDE.append(dt_f)

        torch.cuda.synchronize()
        tin_bc=time.time()
        loss_bc  = pde.loss_BC(data.X_bc)
        torch.cuda.synchronize()
        tout_bc=time.time()
        dt_bc = tout_bc-tin_bc
        listTimeBC.append(dt_bc)

        # loss =  loss_f + loss_bc
        # loss.backward(retain_graph=True)
        # pde.optimizer.step()
        # pde.scheduler.step()
    meanTimePDE = np.mean(listTimePDE[k:])
    # print(f"==>> TimePDE: {listTimePDE}")
    meanTimeBC = np.mean(listTimeBC[k:])
    # print(f"==>> TimeBC: {listTimeBC}")
    return meanTimePDE, meanTimeBC