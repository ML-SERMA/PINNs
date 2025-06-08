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
from pyPINNs.PDE.mixed_one_group_diffusion_source import mixed_one_group_diffusion_source
from pyPINNs.Mesh.CartesianMesh import CartesianMesh
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.basic_utils import check_create_dir
from pyPINNs.Model.neuron_network.FCN import FCN, FCN_FF
from pyPINNs.Data.DataSet import DataSet
from pyPINNs.Train.train_model import train
results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")

parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='VACUUM_Mixed_FCN_IAEA_5S_SLR1e-3_2000_095_HBC')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[96,86])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=200000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[2]+5*[64]+[3])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Tanh')
parser.add_argument('-s','--sampling', type=str, help="sampling method",default='random')
parser.add_argument('-v', '--verbose',action='count', default=0)  

args = parser.parse_args()

name_folder = args.test + '_nc' + str(args.n_collocation) + '_nb' + str(args.n_boundary) \
                + '_nt' + str(args.n_test) + '_ns' + str(args.n_step) + '_nn' + str(args.n_neuron)+'_'+str(args.activation)+'_'+str(args.sampling)
save_dir = check_create_dir(results_dir+name_folder+'/')

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[96, 86],'boundary_conditions': [['VACUUM', 'VACUUM'],['VACUUM', 'VACUUM']],'type_of_points':args.sampling}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,'resampling_every':0} # 625-64

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'mixed_vacuum_96_86'}


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




model_fcn = FCN_FF(**params_model).to(device)


summary(model_fcn, input_size=(2,))
print('model',model_fcn)

# input('Enter')

# Training Time
mypde = mixed_one_group_diffusion_source(pdeDomain,model_fcn,params_pde,device)

optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

train(mypde,pdeData,**params_train)


# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))

phi_REL_L2, phi_AE, phi_pred, phi_test,_,_ = mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)

# phi_REL_L2 = phi_REL_L2.cpu().detach().numpy()
# phi_AE = phi_AE.cpu().detach().numpy()
# phi_pred = phi_pred.cpu().detach().numpy()
# phi_test = phi_test.cpu().detach().numpy()

# Visualization Flux
visualization.viewErrorAndSolution(params_domain,[96,86],phi_AE,phi_pred,phi_test,save_dir)
visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=True,Error_p=True)

# Visualization currents
show_currents = True
if show_currents:
    p_REL_L2 ,p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=False)
    visualization.viewErrorAndCurrents(params_domain,[96,86],p_AE,p_pred,p_test,save_dir)