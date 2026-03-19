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
from pyPINNs.PDE_dev.DDM_mixed_one_group_diffusion_source import DDM_mixed_one_group_diffusion_source
from pyPINNs.Mesh.CartesianMesh import CartesianMesh
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.basic_utils import check_create_dir
from pyPINNs.Model.neuron_network.FCN import FCN, FCN_FF
from pyPINNs.Data.DataSet import DataSet
from pyPINNs.Train.train_model import train
from pyPINNs.Train.train_DDM import update_boundary, train_DDM

results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")

parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='DDM_2x1_Mixed_FCN_Constant_D1_ns100_it2000_LR_1m3_theta_0.5_0.5')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=5*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="an integer number",default=[60,120])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=1)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="an integer number",default=[2]+2*[64]+[3])
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

nsubdo = 2
params_DDM = {'nsubdo':nsubdo, 'gridDDM':[2,1]}


params_domain = [{'xmin':[0, 0], 'xmax':[0.5, 1],'boundary_conditions': [['ZERO_FLUX', 'INTERFACE'],['ZERO_FLUX', 'ZERO_FLUX']]},
                {'xmin':[0.5, 0], 'xmax':[1, 1],'boundary_conditions': [['INTERFACE', 'ZERO_FLUX'],['ZERO_FLUX', 'ZERO_FLUX']]},
                ]

params_data = nsubdo*[{'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,'random_seed':2024,'device':device}]
params_train = [{'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss_0.csv','save_dir':save_dir,
                    'learning_rate_annealing':False,'mode':'a'},
                {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss_1.csv','save_dir':save_dir,
                    'learning_rate_annealing':False,'mode':'a'}
                ]
params_model = nsubdo*[{'layers':args.n_neuron,'activation':args.activation,'device':device,'fourier_mapping_size':None,'hard_BC': 'mixed_L1'}]


def func_D(X):
    return torch.ones_like(X[:,0:1])
def func_Sigma_a(X):
    return torch.ones_like(X[:,0:1])
def func_Source(X):
    return torch.ones_like(X[:,0:1])

params_pde = nsubdo*[{'func_D':func_D,'func_Sigma_a':func_Sigma_a,'func_Source':func_Source}]

#==========================================================================================================



subdo_domain = [SquareDomain(**params_domain[iDD]) for iDD in range(nsubdo)]
subdo_data = [DataSet(domain=subdo_domain[iDD],**params_data[iDD]) for iDD in range(nsubdo)]
subdo_model_fcn = [FCN_FF(**params_model[iDD]).to(device) for iDD in range(nsubdo)]


##=========================================================
params_main_domain = {'xmin':[0., 0.0], 'xmax':[1, 1]}
params_main_data   = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':[120,120],'random_seed':2024,'device':device}
pdeDomain = SquareDomain(**params_main_domain)
pdeData  = DataSet(domain=pdeDomain,**params_main_data)
pdeData.generate_phi_test(load_dir = data_dir+'Constant_D1/RefFlux120')
pdeData.generate_p_test(load_dir = data_dir+'Constant_D1/RefCurrents120')
##==========================================================

for iDD in range(nsubdo):
    summary(subdo_model_fcn[iDD], input_size=(2,))
    print('model',subdo_model_fcn[iDD])

subdo_pde = [DDM_mixed_one_group_diffusion_source(subdo_domain[iDD],subdo_model_fcn[iDD],params_pde[iDD],device) for iDD in range(nsubdo)]

subdo_optimizer = [torch.optim.Adam(subdo_pde[iDD].model.parameters(), lr=1.e-3,weight_decay=0.0) for iDD in range(nsubdo)]
subdo_scheduler = [torch.optim.lr_scheduler.MultiStepLR(subdo_optimizer[iDD], milestones=[50000,100000,200000,300000,400000,500000], gamma=0.2) for iDD in range(nsubdo)]
for iDD in range(nsubdo):
    subdo_pde[iDD].compile(optimizer=subdo_optimizer[iDD],scheduler=subdo_scheduler[iDD])



# # boundary
subdo_pde[0].g_bc = [[None, None],[None,None]]
subdo_pde[1].g_bc = [[None,None],[None,None]]



train_DDM(gridDDM=[2,1],subdo_pde=subdo_pde,subdo_data=subdo_data,params_train=params_train,n_step=20000)



for iDD in range(nsubdo):
    phi_pred_iDD = subdo_pde[iDD].phi_predict(subdo_data[iDD].X_test)
    visualization.viewSolution(params_domain[iDD],[60,120],phi_pred_iDD,save_dir+'phi_'+str(iDD)+'.png')
    visualization.show_loss(pathFile=save_dir+'Loss_'+ str(iDD) +'.csv',save_dir=save_dir+'_domain_'+str(iDD)+'_',Error_phi=False,Error_p=False)


# On the main domain

phi_pred = (pdeData.X_test[:,0:1]<=0.5)*subdo_pde[0].model.forward(pdeData.X_test)[:,0:1] + (pdeData.X_test[:,0:1]>0.5)*subdo_pde[1].model.forward(pdeData.X_test)[:,0:1] 
visualization.viewSolution(params_main_domain,[120,120],phi_pred,save_dir+'phi'+'.png')


phi_test = pdeData.phi_test
phi_AE= torch.abs(phi_pred- phi_test)
visualization.viewErrorAndSolution(params_main_domain,[120,120],phi_AE,phi_pred,phi_test,save_dir)