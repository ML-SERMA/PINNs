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
from pyPINNs.PDE.mixed_two_group_diffusion_source import mixed_two_group_diffusion_source
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
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_FCN_2G_C5G7_L1_case1_SLR1e-3_5000_095_HBC')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[120,120])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=800000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[2]+5*[64]+[6])
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



params_domain = {'xmin':[0, 0], 'xmax':[1.0, 1.0],'boundary_conditions': [['REFLECTION', 'ZERO_FLUX'],['REFLECTION', 'ZERO_FLUX']]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,
                'resampling_every':0, 'ngroup':2}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'2G_C5G7_L1'}



def func_D1(X):
    x=X[:,0:1]
    y=X[:,1:2]

    D = 1.2/(64.26**2)*torch.ones_like(x) 
  
    return D


def func_D2(X):
    x=X[:,0:1]
    y=X[:,1:2]

    D = 0.2/(64.26**2)*torch.ones_like(x) # Material 3
    D[((x>=0)&(x<=1.0/3)&(y>=0)&(y<=1.0/3))+((x>=1.0/3)&(x<=2.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.4/(64.26**2) # Material 1
    D[((x>=1.0/3)&(x<=2.0/3)&(y>=0)&(y<=1.0/3))+((x>=0)&(x<=1.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.4/(64.26**2) # Material 2
   
    return D

def func_Sigma_a1(X):
    x=X[:,0:1]
    y=X[:,1:2]
    Sigma_a = 0.001*torch.ones_like(x)
    Sigma_a[((x>=0)&(x<=1.0/3)&(y>=0)&(y<=1.0/3))+((x>=1.0/3)&(x<=2.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.015 # Material 1
    Sigma_a[((x>=1.0/3)&(x<=2.0/3)&(y>=0)&(y<=1.0/3))+((x>=0)&(x<=1.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.015 # Material 2
    return Sigma_a

def func_Sigma_a2(X):
    x=X[:,0:1]
    y=X[:,1:2]
    Sigma_a = 0.04*torch.ones_like(x)
    Sigma_a[((x>=0)&(x<=1.0/3)&(y>=0)&(y<=1.0/3))+((x>=1.0/3)&(x<=2.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.3 # Material 1
    Sigma_a[((x>=1.0/3)&(x<=2.0/3)&(y>=0)&(y<=1.0/3))+((x>=0)&(x<=1.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.25 # Material 2
    return Sigma_a



def func_Sigma_s12(X):
    x=X[:,0:1]
    y=X[:,1:2]
    Sigma_s = 0.05*torch.ones_like(x)
    Sigma_s[((x>=0)&(x<=1.0/3)&(y>=0)&(y<=1.0/3))+((x>=1.0/3)&(x<=2.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.015 # Material 1
    Sigma_s[((x>=1.0/3)&(x<=2.0/3)&(y>=0)&(y<=1.0/3))+((x>=0)&(x<=1.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.015 # Material 2
    return Sigma_s




def func_S1(X):
    x=X[:,0:1]
    y=X[:,1:2]
    S = torch.zeros_like(x)
    S[((x>=0)&(x<=1.0/3)&(y>=0)&(y<=1.0/3))+((x>=1.0/3)&(x<=2.0/3)&(y>=1.0/3)&(y<=2.0/3))] = 0.0075 #0.45 # Material 1
    S[((x>=1.0/3)&(x<=2.0/3)&(y>=0)&(y<=1.0/3))+((x>=0)&(x<=1.0/3)&(y>=1.0/3)&(y<=2.0/3))]= 0.0075  #0.375 # Material 2
    return S




def func_S2(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    S = torch.zeros_like(x)
    return S



params_pde = {'func_D1':func_D1,'func_D2':func_D2,'func_Sigma_a1':func_Sigma_a1,'func_Sigma_a2':func_Sigma_a2,'func_Sigma_s12':func_Sigma_s12,
            'func_S1':func_S1,'func_S2':func_S2}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'C5G7/RefFlux120_case1',multigroup=True)
pdeData.generate_p_test(load_dir = data_dir+'C5G7/RefCurrents120_case1',multigroup=True)





visualization.viewCrossSection(pdeData.X_train,func_D1(pdeData.X_train),pathFile=save_dir+'D1.png')
visualization.viewCrossSection(pdeData.X_train,func_D2(pdeData.X_train),pathFile=save_dir+'D2.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_a1(pdeData.X_train),pathFile=save_dir+'Sigma_a1.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_a2(pdeData.X_train),pathFile=save_dir+'Sigma_a2.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_s12(pdeData.X_train),pathFile=save_dir+'Sigma_s12.png')
visualization.viewCrossSection(pdeData.X_train,func_S1(pdeData.X_train),pathFile=save_dir+'S1.png')

# input('Enter')

model_fcn = FCN_FF(**params_model).to(device)
summary(model_fcn, input_size=(2,))
print('model',model_fcn)



# Training Time
mypde = mixed_two_group_diffusion_source(pdeDomain,model_fcn,params_pde,device)

# parameters = [*mypde.model.parameters(), *mypde.loss_function.parameters()]
# optimizer = torch.optim.Adam(parameters, lr=1.e-3,weight_decay=0.0)

optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.95)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

train(mypde,pdeData,**params_train)


# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))
solution = mypde.full_predict(pdeData.X_test)


phi_pred = [solution[:,0:1],solution[:,3:4]]

ngroup =2
for igroup in range(ngroup):
    visualization.viewSolution(params_domain,[120,120],phi_pred[igroup],save_dir+'flux_group_'+str(igroup)+'.png')

visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=True,Error_p=True)

phi_REL_L2, phi_AE, phi_pred, phi_test,_ = mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False,ngroup=2)

for igroup in range(ngroup):
    visualization.viewErrorAndSolution(params_domain,[120,120],phi_AE[igroup],phi_pred[igroup],phi_test[igroup],
                                    save_dir,'error_phi_group_'+str(igroup)+'.png')

# Visualization currents
show_currents = True
if show_currents:
    for igroup in range(ngroup):
        p_REL_L2 ,p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=False,ngroup=2)
        visualization.viewErrorAndCurrents(params_domain,[120,120],p_AE[igroup],p_pred[igroup],p_test[igroup],
                                            save_dir,'error_current_group_'+str(igroup)+'.png')
