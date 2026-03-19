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
from pyPINNs.PDE.mixed_multigroup_diffusion_eigenvalue import mixed_multigroup_diffusion_eigenvalue
from pyPINNs.Geometry.CartesianGeometry import CartesianGeometry
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.basic_utils import check_create_dir
from pyPINNs.Model.neuron_network.FCN import FCN, FCN_FF
from pyPINNs.Data.DataSet import DataSet
from pyPINNs.Train.train_model import train
from pyPINNs.Tools.lr_scheduler import StepLRWithDecayingRestart
from pyPINNs.Tools.attention_loss import IRDR_BRDRLayer
results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")

parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='EP_Mixed_FCN_pool_type_SLR2e-4_09_Nit2000_HBC')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[96,86])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=2000000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[2]+5*[64]+[3])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Sin')
parser.add_argument('-s','--sampling', type=str, help="sampling method",default='Sobol')
parser.add_argument('-v', '--verbose',action='count', default=0)
parser.add_argument("--scaling-loss",dest="scaling_loss",action="store_true",default=True,help="Enable loss scaling")
parser.add_argument("--no-scaling-loss",dest="scaling_loss",action="store_false",help="Disable loss scaling")



args = parser.parse_args()
scaling_tag = "Scaled" if args.scaling_loss else "Unscaled"
name_folder = args.test + '_nc' + str(args.n_collocation) + '_nb' + str(args.n_boundary) \
                + '_nt' + str(args.n_test) + '_ns' + str(args.n_step) + '_nn' + str(args.n_neuron)\
                +'_'+str(args.activation)+'_'+str(args.sampling) + "_" + scaling_tag
                
save_dir = check_create_dir(results_dir+name_folder+'/')

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(1)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[96, 86],'boundary_conditions': [['VACUUM', 'VACUUM'],['VACUUM', 'VACUUM']],'type_of_points':args.sampling}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,
                'resampling_every':0,'normalization':True} 

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'mixed_vacuum_96_86'}

params_solver = {'scaling_loss':args.scaling_loss,
                'momentum':False,'beta1':0.0,'beta2':0.6,
                'num_inner_iters':2000, 'keff_ref':1.00693,'verbose':2, 'save_dir':save_dir,
                'anderson':False,'beta':0.8,'m':4}

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

    return D # Shape: (N, 1)


def func_Sigma_r(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    Sigma_r = 0.01*torch.ones_like(x)
    # R1
    Sigma_r[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 0.07
    # R2 
    Sigma_r[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 0.28
    # R3
    Sigma_r[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 0.04
    # R4 
    Sigma_r[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 0.15

    return Sigma_r # Shape: (N, 1)

def func_Sigma_s(X):
    N = X.shape[0]
    G = 1  # Two groups
    Sigma_s = torch.zeros(N, G, G, device=X.device)  # [N, G, G]

    return Sigma_s  # shape (N, G, G)


def func_NuSigma_f(X):
    x=X[:,0:1]
    y=X[:,1:2]
    # R5 
    Sigma_f = 0.0*torch.ones_like(x)
    # R1
    Sigma_f[(x>=18)&(x<=48)&(y>=18)&(y<=43)] = 0.079
    # R2 
    Sigma_f[(x>=48)&(x<=78)&(y>=18)&(y<=43)] = 0.0
    # R3
    Sigma_f[(x>=48)&(x<=78)&(y>=43)&(y<=68)] = 0.043
    # R4 
    Sigma_f[(x>=18)&(x<=48)&(y>=43)&(y<=68)] = 0.0

    return Sigma_f

chi = torch.tensor([1.])

params_pde = {'func_D':func_D,'func_Sigma_r':func_Sigma_r,'func_Sigma_s':func_Sigma_s,
                'func_NuSigma_f':func_NuSigma_f,'chi':chi, 'num_groups':1}


pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'EP_IAEA-5S/RefFlux_96_86')
pdeData.generate_p_test(load_dir = data_dir+'EP_IAEA-5S/RefCurrents_96_86')



visualization.viewCrossSection(pdeData.X_train,func_D(pdeData.X_train),pathFile=save_dir+'D.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_r(pdeData.X_train),pathFile=save_dir+'Sigma_r.png')
visualization.viewCrossSection(pdeData.X_train,func_NuSigma_f(pdeData.X_train),pathFile=save_dir+'NuSigma_f.png')




model_fcn = FCN_FF(**params_model).to(device)
summary(model_fcn, input_size=(2,))
print('model',model_fcn)

# input('Enter')

# Training Time
mypde = mixed_multigroup_diffusion_eigenvalue(pdeDomain,model_fcn,params_pde,params_solver,device)


optimizer = torch.optim.Adam(mypde.model.parameters(), lr=2.e-4,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

# resampler = Resampler(mypde,pdeData,save_dir=save_dir,interval=6e4,min_iter=4e4,max_iter=4e5,method='EVO')
# IRDR_attention = IRDR_BRDRLayer(mode="all")

train(mypde,pdeData,**params_train,attention_loss=None)


# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))
solution = mypde.full_predict(pdeData.X_test)

phi_pred= mypde.unpack_solution(solution)[0] 

mesh = CartesianGeometry(ndim=2,xmin=params_domain['xmin'],xmax=params_domain['xmax'],num_cells=args.n_test)
visualization.export_flux_list(mesh,phi_pred,filename= save_dir+'flux_data.vtr')
visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,n_phi_groups=1,n_p_groups=1)


##======================================================================================================
ngroup =1
for igroup in range(ngroup):
    visualization.viewFlux(mesh,phi_pred[igroup],save_dir,'flux_group_'+str(igroup)+'.png',index='xy')

show_flux = True
if show_flux:
    phi_AE, phi_pred, phi_test, mass_phi_pred, mass_phi_test= mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=True)[2:]
    for igroup in range(ngroup):
        visualization.viewErrorAndFlux(mesh,phi_AE[igroup],phi_pred[igroup],phi_test[igroup],
                                        save_dir,'error_phi_group_'+str(igroup)+'.png',index='xy')

show_currents = True
if show_currents:
    p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=True,
                                                   mass_pred=mass_phi_pred,mass_test=mass_phi_test)[2:]
    for igroup in range(ngroup):
        visualization.viewErrorAndCurrents(mesh,p_AE[igroup],p_pred[igroup],p_test[igroup],
                                            save_dir,'error_current_group_'+str(igroup)+'.png',index='xy')

visualization.show_history_residuals(pathFile=save_dir+'train.csv',save_dir=save_dir,keff_ref_exist=True)


