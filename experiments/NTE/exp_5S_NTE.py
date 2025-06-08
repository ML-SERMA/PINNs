import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from torchsummary import summary

project_dir  = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(str(project_dir))

from pyPINNs.Domain.squareShape import SquareDomain
from pyPINNs.PDE.neutron_transport_equation import NeutronTransportMultiOutput
from pyPINNs.Mesh.CartesianMesh import CartesianMesh
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.basic_utils import check_create_dir
from pyPINNs.Model.neuron_network.FCN import FCN_FF
from pyPINNs.Data.DataSet import DataSet
from pyPINNs.Train.train_model import train
results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")

parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='5S_NTE_SLR1e-3_2000_095_SBC_new')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[96,86])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=200000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[4]+5*[64]+[1])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Tanh')
parser.add_argument('-v', '--verbose',action='count', default=0)  

args = parser.parse_args()

name_folder = args.test + '_nc' + str(args.n_collocation) + '_nb' + str(args.n_boundary) \
                + '_nt' + str(args.n_test) + '_ns' + str(args.n_step) + '_nn' + str(args.n_neuron)+'_'+str(args.activation)
save_dir = check_create_dir(results_dir+name_folder+'/')

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[96, 86]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,
                'resampling_every':0, 'ngroup':1}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':None}


def Sigma_t_func(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    Sigma_t = 0.9*torch.ones_like(x)
    # R1
    Sigma_t[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 0.6
    # R2 
    Sigma_t[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 0.48
    # R3
    Sigma_t[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 0.7
    # R4 
    Sigma_t[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 0.65
    return Sigma_t


def Sigma_s_func(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    Sigma_s = 0.89*torch.ones_like(x)
    # R1
    Sigma_s[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 0.53
    # R2 
    Sigma_s[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 0.2
    # R3
    Sigma_s[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 0.66
    # R4 
    Sigma_s[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 0.5

    return Sigma_s.unsqueeze(-1)

def q_func(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    Source = torch.zeros_like(x)
    # R1
    Source[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 1.0
    # R2 
    Source[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 0.0
    # R3
    Source[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 1.0
    # R4 
    Source[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 0.0
    return Source




params_pde = {'Sigma_t_func':Sigma_t_func,'Sigma_s_func':Sigma_s_func,'q_func':q_func}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
# pdeData.generate_phi_test(load_dir = data_dir+'Center2G/RefFlux140',multigroup=True)
# pdeData.generate_p_test(load_dir = data_dir+'Center2G/RefCurrents140',multigroup=True)
# pdeData.generate_phi_test(load_dir = data_dir+'IAEA-5S/RefFlux_96_86')
# pdeData.generate_p_test(load_dir = data_dir+'IAEA-5S/RefCurrents_96_86')



# 2D spatial + 2D angular (so input is [x, y, mu, eta])
n_quad = 8 # Q = n_quad
dim_angular = 2
num_groups = 1 # G

input_dim = 2 + 2  # [x, y, mu, eta]
output_dim = num_groups

model_fcn = FCN_FF(**params_model).to(device)
summary(model_fcn, input_size=(input_dim,))
print('model',model_fcn)



# Training Time
mypde = NeutronTransportMultiOutput(pdeDomain,model_fcn,params_pde,n_quad=n_quad,dim_angular=dim_angular,
                                    num_groups=num_groups, isotropic_kernel=True,device=device)
optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

# res_pde = mypde.residual_PDE(pdeData.X_train)
# loss = torch.mean(res_pde**2)
# print(f"==>> loss: {loss}")
# res_bc = mypde.residual_BC(pdeData.X_bc)
# print(f"==>> res_bc: {res_bc.shape}")


train(mypde,pdeData,**params_train)

# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))


scalar_flux = mypde.compute_scalar_flux(pdeData.X_test)


phi_pred = scalar_flux

for igroup in range(num_groups):
    visualization.viewSolution(params_domain,[96,86],phi_pred[:,igroup],save_dir+'flux_group_'+str(igroup)+'.png')
    # visualization.viewErrorAndSolution(params_domain,[96,86],phi_AE,phi_pred,phi_test,save_dir)

visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=False,Error_p=False)




