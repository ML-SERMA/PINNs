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
from pyPINNs.PDE.mixed_diffusion_one_group import mixed_diffusion_one_group
from pyPINNs.PDE.primal_diffusion_one_group import primal_diffusion_one_group
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
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_IAEA_5S')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test', nargs='+', type=int, help="an integer number",default=[96,86])
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



params_domain = {'xmin':[0, 0], 'xmax':[96, 86],'boundary_conditions': [['VACUUM', 'VACUUM'],['VACUUM', 'VACUUM']]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,'random_seed':2024,'device':device}
params_model = {'layers': [2]+5*[64]+[3],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':'mixed_vacuum_96_86'}
# params_model_f = {'layers': [2]+5*[64]+[3],'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC':'mixed_vacuum_96_86'}

def func_D(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    D = (1.0/2.7)*torch.ones_like(x)
    # R1
    D[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 1.0/1.8
    # R2 
    D[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 1.0/1.44
    # R3
    D[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 1.0/2.1
    # R4 
    D[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 1.0/1.95

    return D


def func_Sigma_a(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    Sigma_a = 0.01*torch.ones_like(x)
    # R1
    Sigma_a[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 0.07
    # R2 
    Sigma_a[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 0.28
    # R3
    Sigma_a[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 0.04
    # R4 
    Sigma_a[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 0.15

    return Sigma_a

def func_Source(X):
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

params_pde = {'func_D':func_D,'func_Sigma_a':func_Sigma_a,'func_Source':func_Source}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'IAEA-5S/RefFlux_96_86')
pdeData.generate_p_test(load_dir = data_dir+'IAEA-5S/RefCurrents_96_86')



visualization.viewCrossSection(pdeData.X_train,func_D(pdeData.X_train),pathFile=save_dir+'D.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_a(pdeData.X_train),pathFile=save_dir+'Sigma_a.png')
visualization.viewCrossSection(pdeData.X_train,func_Source(pdeData.X_train),pathFile=save_dir+'Source.png')



model_fcn_c  = FCN_FF(**params_model).to(device)
model_fcn_f = FCN_FF(**params_model).to(device)

summary(model_fcn_c, input_size=(2,))
summary(model_fcn_f, input_size=(2,))




# Training Time
mypde_c  = mixed_diffusion_one_group(pdeDomain,model_fcn_c,params_pde,device)
mypde_f = mixed_diffusion_one_group(pdeDomain,model_fcn_f,params_pde,device)

# optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
# mypde.compile(optimizer=optimizer,scheduler=scheduler)

# train(mypde,pdeData,**params_train)


# Model Accuracy 
mypde_c.model.load_state_dict(torch.load(save_dir+"model_nr10240.pt"))
mypde_f.model.load_state_dict(torch.load(save_dir+"model_nr20480.pt"))

phi_REL_L2_c, phi_AE_c, phi_pred_c, phi_test_c,_ = mypde_c.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)
phi_REL_L2_f, phi_AE_f, phi_pred_f, phi_test_f,_ = mypde_f.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)

minE = torch.min(torch.stack([torch.min(phi_AE_c),torch.min(phi_AE_f)]))
print(f"==>> minE: {minE}")
maxE = torch.max(torch.stack([torch.max(phi_AE_c),torch.max(phi_AE_f)]))
print(f"==>> maxE: {maxE}")

minF = torch.min(torch.stack([torch.min(phi_pred_c),torch.min(phi_pred_f),torch.min(phi_test_c)]))
print(f"==>> minF: {minF}")
maxF = torch.max(torch.stack([torch.max(phi_pred_c),torch.max(phi_pred_f),torch.max(phi_test_c)]))
print(f"==>> maxF: {maxF}")





# Visualization Flux
visualization.viewErrorAndSolution(params_domain,[96,86],phi_AE_c,phi_pred_c,phi_test_c,save_dir,
                                   name='error_phi_test_10240.png',vminE=minE,vmaxE=maxE,vminF=minF,vmaxF=maxF)
visualization.viewErrorAndSolution(params_domain,[96,86],phi_AE_f,phi_pred_f,phi_test_f,save_dir,
                                   name='error_phi_test_20480.png',vminE=minE,vmaxE=maxE,vminF=minF,vmaxF=maxF)

gapError = torch.abs(phi_pred_c-phi_pred_f)
visualization.viewSolution(params_domain,[96,86],gapError,pathFile=save_dir+'gapError.png')

# p_REL_L2_mixed ,p_AE_mixed, p_pred_mixed, p_test_mixed = mypde_mixed.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=False)
# visualization.viewErrorAndCurrents(params_domain,[120,120],p_AE_mixed,p_pred_mixed,p_test_mixed,save_dir,name='error_current_test_mixed.png')

