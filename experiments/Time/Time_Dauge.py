import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from torchsummary import summary
from tqdm import tqdm
import time

project_dir  = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(str(project_dir))

from pyPINNs.Domain.squareShape import SquareDomain
from pyPINNs.PDE.mixed_one_group_diffusion_source import mixed_one_group_diffusion_source
from pyPINNs.PDE.primal_one_group_diffusion_source import primal_one_group_diffusion_source
from pyPINNs.Mesh.CartesianMesh import CartesianMesh
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.basic_utils import check_create_dir
from pyPINNs.Tools.utils import get_time, get_time_gpu
from pyPINNs.Model.neuron_network.FCN import FCN_FF
from pyPINNs.Data.DataSet import DataSet
from pyPINNs.Train.train_model import train
results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")

parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_Primal_Dauge_D10')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test', nargs='+', type=int, help="an integer number",default=[120,120])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=200)
# parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
# parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="an integer number",default=[2]+5*[64]+[3])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Tanh')
parser.add_argument('-v', '--verbose',action='count', default=0)  

args = parser.parse_args()

name_folder = args.test 
save_dir = check_create_dir(results_dir+name_folder+'/')

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
# device = torch.device('cpu')    
print(f"Running on {device}. Yes! ")



# params_domain = {'xmin':[0, 0], 'xmax':[1, 1],'boundary_conditions':[['ZERO_FLUX', 'ZERO_FLUX'],['ZERO_FLUX', 'ZERO_FLUX']]}
params_domain = {'xmin':[0, 0], 'xmax':[1, 1],'boundary_conditions': [['VACUUM', 'VACUUM'],['VACUUM', 'VACUUM']]}

params_data = [{'n_collocation':10*1024,'n_boundary':512,'n_test':args.n_test,'random_seed':2024,'device':device},
               {'n_collocation':40*1024,'n_boundary':10*512,'n_test':args.n_test,'random_seed':2024,'device':device},
               {'n_collocation':80*1024,'n_boundary':20*512,'n_test':args.n_test,'random_seed':2024,'device':device},
                 ]
params_model_mixed = [{'layers': [2]+3*[64]+[3],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':None},
                      {'layers': [2]+5*[64]+[3],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':None},
                      {'layers': [2]+7*[64]+[3],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':None},
                        ]
params_model_primal = [{'layers': [2]+3*[64]+[1],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':None},
                       {'layers': [2]+5*[64]+[1],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':None},
                       {'layers': [2]+7*[64]+[1],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':None}
                       ]

def func_D(X):
    Dl = 1.0
    Dh = 10.0

    x=X[:,0:1]
    y=X[:,1:2]
    D = Dl*torch.ones_like(x)
    D[(x>=0.5)&(y<=0.5)] = Dh
    D[(x<=0.5)&(y>=0.5)] = Dh
    return D

def func_Sigma_a(X):
    return torch.ones_like(X[:,0:1])

def func_Source(X):
    return torch.ones_like(X[:,0:1])


params_pde = {'func_D':func_D,'func_Sigma_a':func_Sigma_a,'func_Source':func_Source}





pdeDomain = SquareDomain(**params_domain) 
pdeData = [DataSet(domain=pdeDomain,**params_data[id]) for id in range(len(params_data))]
model_fcn_mixed  = [FCN_FF(**params_model_mixed[id]).to(device) for id in range(len(params_model_mixed))]
model_fcn_primal = [FCN_FF(**params_model_primal[id]).to(device) for id in range(len(params_model_primal))]

pde_mixed  = [mixed_one_group_diffusion_source(pdeDomain,model_fcn_mixed[id],params_pde,device) for id in range(len(model_fcn_mixed))]
pde_primal = [primal_one_group_diffusion_source(pdeDomain,model_fcn_primal[id],params_pde,device) for id in range(len(model_fcn_primal))]

# optimizer_mixed = [torch.optim.Adam(pde_mixed[id].model.parameters(), lr=1.e-3,weight_decay=0.0) for id in range(2)]
# scheduler_mixed = [torch.optim.lr_scheduler.StepLR(optimizer_mixed[id], step_size=2000, gamma=0.95) for id in range(2)]
# for id in range(2):
#     pde_mixed[id].compile(optimizer=optimizer_mixed[id],scheduler=scheduler_mixed[id])

# optimizer_primal = [torch.optim.Adam(pde_primal.model.parameters(), lr=1.e-3,weight_decay=0.0) for id in range(2)]
# scheduler_primal = [torch.optim.lr_scheduler.StepLR(optimizer_primal, step_size=2000, gamma=0.95) for id in range(2)]
# for id in range(2):
#     pde_primal[id].compile(optimizer=optimizer_primal[id],scheduler=scheduler_primal[id])




my_PDE_primal = []
my_BC_primal = []
my_PDE_mixed = []
my_BC_mixed  = []
for id in range(len(pdeData)):
    list_PDE_primal = []
    list_BC_primal  = []
    list_PDE_mixed = []
    list_BC_mixed = []
    for im in range(len(pde_mixed)):
        # print(f'id= {id}, im = {im}')
        
        time_PDE_primal, time_BC_primal = get_time_gpu(pde_primal[im],pdeData[id],n_step=1000,k=2)
        time_PDE_mixed, time_BC_mixed = get_time_gpu(pde_mixed[im],pdeData[id],n_step=1000,k=2)
        list_PDE_primal.append(time_PDE_primal)
        list_BC_primal.append(time_BC_primal)
        list_PDE_mixed.append(time_PDE_mixed)
        list_BC_mixed.append(time_BC_mixed)
        # print(f"==>> time_PDE_primal: {time_PDE_primal}")
        # print(f"==>> time_BC_primal: {time_BC_primal}")
        # print(f"==>> time_PDE_mixed: {time_PDE_mixed}")
        # print(f"==>> time_BC_mixed: {time_BC_mixed}")
        
    arr_PDE_primal = np.array(list_PDE_primal)
    arr_BC_primal  = np.array(list_BC_primal)
    arr_PDE_mixed  = np.array(list_PDE_mixed)
    arr_BC_mixed   = np.array(list_BC_mixed)

    my_PDE_primal.append(arr_PDE_primal)
    my_BC_primal.append(arr_BC_primal)
    my_PDE_mixed.append(arr_PDE_mixed)
    my_BC_mixed.append(arr_BC_mixed)
PDE_primal = np.vstack(my_PDE_primal)
BC_primal  =np.vstack(my_BC_primal)
PDE_mixed = np.vstack(my_PDE_mixed)
BC_mixed   = np.vstack(my_BC_mixed)
print(f"==>> PDE_primal: {PDE_primal}")
print(f"==>> PDE_mixed: {PDE_mixed}")

print(f"==>> BC_primal: {BC_primal}")
print(f"==>> BC_mixed: {BC_mixed}")









# pdeDomain = SquareDomain(**params_domain)
# pdeData = DataSet(domain=pdeDomain,**params_data)
# model_fcn_mixed  = FCN_FF(**params_model_mixed).to(device)
# model_fcn_primal = FCN_FF(**params_model_primal).to(device)
# pde_mixed  = mixed_diffusion_one_group(pdeDomain,model_fcn_mixed,params_pde,device)
# pde_primal = primal_diffusion_one_group(pdeDomain,model_fcn_primal,params_pde,device)

# optimizer_mixed = torch.optim.Adam(pde_mixed.model.parameters(), lr=1.e-3,weight_decay=0.0)
# scheduler_mixed = torch.optim.lr_scheduler.StepLR(optimizer_mixed, step_size=2000, gamma=0.95)
# pde_mixed.compile(optimizer=optimizer_mixed,scheduler=scheduler_mixed)

# optimizer_primal = torch.optim.Adam(pde_primal.model.parameters(), lr=1.e-3,weight_decay=0.0)
# scheduler_primal = torch.optim.lr_scheduler.StepLR(optimizer_primal, step_size=2000, gamma=0.95)
# pde_primal.compile(optimizer=optimizer_primal,scheduler=scheduler_primal)



# listTimePDE_mixed = []
# listTimeBC_mixed  = []

# for it in tqdm(range(10)):
#     pde_mixed.optimizer.zero_grad()

#     tin_f=time.time()
#     loss_f_mixed   = pde_mixed.loss_PDE(pdeData.X_train)
#     tout_f=time.time()
#     dt_f = tout_f-tin_f
#     listTimePDE_mixed.append(dt_f)

#     tin_bc=time.time()
#     loss_bc_mixed  = pde_mixed.loss_BC(pdeData.X_bc)
#     tout_bc=time.time()
#     dt_bc = tout_bc-tin_bc
#     listTimeBC_mixed.append(dt_bc)

#     loss_mixed =  loss_f_mixed + loss_bc_mixed
#     loss_mixed.backward(retain_graph=True)
#     pde_mixed.optimizer.step()
#     pde_mixed.scheduler.step()

# print(f"==>> listTimePDE_mixed: {listTimePDE_mixed}")
# print(f"==>> listTimeBC_mixed: {listTimeBC_mixed}")

# listTimePDE_primal = []
# listTimeBC_primal  = []

# for it in tqdm(range(10)):
#     pde_primal.optimizer.zero_grad()

#     tin_f=time.time()
#     loss_f_primal   = pde_primal.loss_PDE(pdeData.X_train)
#     tout_f=time.time()
#     dt_f = tout_f-tin_f
#     listTimePDE_primal.append(dt_f)

#     tin_bc=time.time()
#     loss_bc_primal  = pde_primal.loss_BC(pdeData.X_bc)
#     tout_bc=time.time()
#     dt_bc = tout_bc-tin_bc
#     listTimeBC_primal.append(dt_bc)

#     loss_primal =  loss_f_primal + loss_bc_primal
#     loss_primal.backward(retain_graph=True)
#     pde_primal.optimizer.step()
#     pde_primal.scheduler.step()

# print(f"==>> listTimePDE_primal: {listTimePDE_primal}")
# print(f"==>> listTimeBC_primal: {listTimeBC_primal}")


















# import matplotlib.pyplot as plt
# fig = plt.figure(figsize =(10, 7))

# # Creating plot
# plt.boxplot(Time_PDE_mixed)
# plt.savefig(save_dir+'mixed.pdf')
# # show plot
# plt.show()

# fig = plt.figure(figsize =(10, 7))

# # Creating plot
# plt.boxplot(Time_PDE_primal)
# plt.savefig(save_dir+'primal.pdf')
# # show plot
# plt.show()