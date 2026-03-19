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
from pyPINNs.PDE_dev.mixed_one_group_diffusion_source import mixed_one_group_diffusion_source
from pyPINNs.PDE_dev.primal_one_group_diffusion_source import primal_one_group_diffusion_source
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
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_Primal_Center_D10')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test', nargs='+', type=int, help="an integer number",default=[120,120])
# parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=200000)
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
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[1, 1]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,'random_seed':2024,'device':device}
params_model_mixed = {'layers': [2]+5*[64]+[3],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':'mixed_L1'}
params_model_primal = {'layers': [2]+5*[64]+[1],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':'mixed_L1'}

def func_D(X):
    Dl = 1.0
    Dh = 10.0
    x=X[:,0:1]
    y=X[:,1:2]
    
    D = Dl*torch.ones_like(x)
    D[(x>=1.0/3)&(x<=2.0/3)&(y>=1.0/3)&(y<=2.0/3)] = Dh
    return D

def func_Sigma_a(X):
    return torch.ones_like(X[:,0:1])

def func_Source(X):
    return torch.ones_like(X[:,0:1])


params_pde = {'func_D':func_D,'func_Sigma_a':func_Sigma_a,'func_Source':func_Source}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'Center_D10/RefFlux120')
pdeData.generate_p_test(load_dir = data_dir+'Center_D10/RefCurrents120')



model_fcn_mixed  = FCN_FF(**params_model_mixed).to(device)
model_fcn_primal = FCN_FF(**params_model_primal).to(device)

summary(model_fcn_mixed, input_size=(2,))
summary(model_fcn_primal, input_size=(2,))




# Training Time
mypde_mixed  = mixed_one_group_diffusion_source(pdeDomain,model_fcn_mixed,params_pde,device)
mypde_primal = primal_one_group_diffusion_source(pdeDomain,model_fcn_primal,params_pde,device)

# optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
# mypde.compile(optimizer=optimizer,scheduler=scheduler)

# train(mypde,pdeData,**params_train)


# Model Accuracy 
mypde_mixed.model.load_state_dict(torch.load(save_dir+"model_mixed.pt"))
mypde_primal.model.load_state_dict(torch.load(save_dir+"model_primal.pt"))

phi_REL_L2_mixed, phi_AE_mixed, phi_pred_mixed, phi_test_mixed,_,_ = mypde_mixed.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)
phi_REL_L2_primal, phi_AE_primal, phi_pred_primal, phi_test_primal,_,_ = mypde_primal.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)

minE = torch.min(torch.stack([torch.min(phi_AE_mixed),torch.min(phi_AE_primal)]))
print(f"==>> minE: {minE}")
maxE = torch.max(torch.stack([torch.max(phi_AE_mixed),torch.max(phi_AE_primal)]))
print(f"==>> maxE: {maxE}")

minF = torch.min(torch.stack([torch.min(phi_pred_mixed),torch.min(phi_pred_primal),torch.min(phi_test_mixed)]))
print(f"==>> minF: {minF}")
maxF = torch.max(torch.stack([torch.max(phi_pred_mixed),torch.max(phi_pred_primal),torch.max(phi_test_mixed)]))
print(f"==>> maxF: {maxF}")





# Visualization Flux
visualization.viewErrorAndSolutionScale(params_domain,[120,120],phi_AE_mixed,phi_pred_mixed,phi_test_mixed,save_dir,
                                name='error_phi_test_mixed.pdf',vminE=minE,vmaxE=maxE,vminF=minF,vmaxF=maxF)
visualization.viewErrorAndSolutionScale(params_domain,[120,120],phi_AE_primal,phi_pred_primal,phi_test_primal,save_dir,
                                name='error_phi_test_primal.pdf',vminE=minE,vmaxE=maxE,vminF=minF,vmaxF=maxF)

# p_REL_L2_mixed ,p_AE_mixed, p_pred_mixed, p_test_mixed = mypde_mixed.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=False)
# visualization.viewErrorAndCurrents(params_domain,[120,120],p_AE_mixed,p_pred_mixed,p_test_mixed,save_dir,name='error_current_test_mixed.png')

