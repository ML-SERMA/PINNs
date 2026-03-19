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
from pyPINNs.PDE.mixed_multigroup_diffusion_source import mixed_multigroup_diffusion_source
from pyPINNs.Geometry.CartesianGeometry import CartesianGeometry
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.resampler import Resampler
from pyPINNs.Tools.basic_utils import check_create_dir
from pyPINNs.Model.neuron_network.FCN import FCN_FF
from pyPINNs.Data.DataSet import DataSet
from pyPINNs.Train.train_model import train
from pyPINNs.Tools.attention_loss import IRDR_BRDRLayer
results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")

parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_FCN_2G_C5G7S_SLR2e-4_2000_095_HBC')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=20*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[120,120])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=400000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[2]+5*[64]+[6])
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
print(f"==>> save_dir: {name_folder}")

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(1)
else:
    device = torch.device('cpu')
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[64.26, 64.26],'boundary_conditions': [['REFLECTION', 'ZERO_FLUX'],['REFLECTION', 'ZERO_FLUX']],
                'type_of_points':args.sampling}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,
                'resampling_every':0, 'ngroup':2}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'2G_C5G7'}

params_solver = {'scaling_loss':args.scaling_loss}


##===========================================================================================================================


def func_D(X):
    x = X[:, 0:1]
    y = X[:, 1:2]

    N = X.shape[0]
    D = torch.zeros(N, 2, device=X.device)  # Shape: (N, G), G = 2

    # --- Material masks ---
    # Material 1
    mask1 = ((x >= 0) & (x <= 21.42) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 21.42) & (x <= 42.84) & (y >= 21.42) & (y <= 42.84))

    # Material 2
    mask2 = ((x >= 21.42) & (x <= 42.84) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 0) & (x <= 21.41) & (y >= 21.42) & (y <= 42.84))

    # Material 3 = everything else (default)

    # --- Assign diffusion coefficients ---
    # Default: Material 3
    D[:, 0] = 1.2  # Group 0
    D[:, 1] = 0.2  # Group 1

    # Overwrite for Materials 1 and 2
    for mask in [mask1, mask2]:
        D[mask.squeeze(), 1] = 0.4  # Group 1 diffusion changes
        # D[:, 0] remains 1.2 everywhere (no change in Group 0)

    return D  # Shape: (N, 2)


def func_Sigma_r(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    N = X.shape[0]
    Sigma_r = torch.zeros(N, 2, device=X.device)  # (N, G)

    # --- Masks for materials ---
    mask1 = ((x >= 0) & (x <= 21.42) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 21.42) & (x <= 42.84) & (y >= 21.42) & (y <= 42.84))

    mask2 = ((x >= 21.42) & (x <= 42.84) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 0) & (x <= 21.41) & (y >= 21.42) & (y <= 42.84))

    # --- Default: Material 3 ---
    Sigma_r[:, 0] = 0.051  # Group 0
    Sigma_r[:, 1] = 0.04   # Group 1

    # --- Material 1 ---
    Sigma_r[mask1.squeeze(), 0] = 0.03
    Sigma_r[mask1.squeeze(), 1] = 0.3

    # --- Material 2 ---
    Sigma_r[mask2.squeeze(), 0] = 0.03
    Sigma_r[mask2.squeeze(), 1] = 0.25

    return Sigma_r  # Shape: (N, G)


def func_Sigma_s(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    N = X.shape[0]
    G = 2  # Two groups

    Sigma_s = torch.zeros(N, G, G, device=X.device)  # [N, G, G]

    # Region masks
    mask1 = ((x >= 0) & (x <= 21.42) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 21.42) & (x <= 42.84) & (y >= 21.42) & (y <= 42.84))
    mask2 = ((x >= 21.42) & (x <= 42.84) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 0) & (x <= 21.41) & (y >= 21.42) & (y <= 42.84))

    # --- Defaults: Material 3 ---
    Sigma_s[:, 0, 0] = 0.0
    Sigma_s[:, 0, 1] = 0.0
    Sigma_s[:, 1, 0] = 0.05   #
    Sigma_s[:, 1, 1] = 0.0

    # --- Material 1 ---
    Sigma_s[mask1.squeeze(), 1, 0] = 0.015  # 
    Sigma_s[mask1.squeeze(), 0, 0] = 0.0
    Sigma_s[mask1.squeeze(), 0, 1] = 0.0
    Sigma_s[mask1.squeeze(), 1, 1] = 0.0

    # --- Material 2 ---
    Sigma_s[mask2.squeeze(), 1, 0] = 0.015  # 
    Sigma_s[mask2.squeeze(), 0, 0] = 0.0
    Sigma_s[mask2.squeeze(), 0, 1] = 0.0
    Sigma_s[mask2.squeeze(), 1, 1] = 0.0

    return Sigma_s  # shape (N, G, G)


def func_Sf(X):
    x = X[:, 0:1]
    y = X[:, 1:2]

    N = X.shape[0]
    G = 2

    S = torch.zeros(N, G, device=X.device)  # [N, G]

    # Region masks
    mask1 = ((x >= 0) & (x <= 21.42) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 21.42) & (x <= 42.84) & (y >= 21.42) & (y <= 42.84))
    mask2 = ((x >= 21.42) & (x <= 42.84) & (y >= 0) & (y <= 21.42)) | \
            ((x >= 0) & (x <= 21.41) & (y >= 21.42) & (y <= 42.84))

    # Group 1 source (nonzero in R1 and R2)
    S[:, 0] = 0.0
    S[mask1.squeeze(), 0] = 0.0075
    S[mask2.squeeze(), 0] = 0.0075

    # Group 2 source remains 0 (as in func_S2)
    return S  # shape [N, 2]

params_pde = {'func_D':func_D,'func_Sigma_r':func_Sigma_r,'func_Sigma_s':func_Sigma_s,'func_Sf':func_Sf, 'num_groups':2}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'C5G7/RefFlux120_case1',multigroup=True)
pdeData.generate_p_test(load_dir = data_dir+'C5G7/RefCurrents120_case1',multigroup=True)





visualization.viewCrossSection(pdeData.X_train,func_D(pdeData.X_train)[:,0],pathFile=save_dir+'D1.png')
visualization.viewCrossSection(pdeData.X_train,func_D(pdeData.X_train)[:,1],pathFile=save_dir+'D2.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_r(pdeData.X_train)[:,0],pathFile=save_dir+'Sigma_r1.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_r(pdeData.X_train)[:,1],pathFile=save_dir+'Sigma_r2.png')
visualization.viewCrossSection(pdeData.X_train,func_Sigma_s(pdeData.X_train)[:,1,0],pathFile=save_dir+'Sigma_s12.png')
visualization.viewCrossSection(pdeData.X_train,func_Sf(pdeData.X_train)[:,0],pathFile=save_dir+'S1.png')

# input('Enter')

model_fcn = FCN_FF(**params_model).to(device)
summary(model_fcn, input_size=(2,))
print('model',model_fcn)



# Training Time
mypde = mixed_multigroup_diffusion_source(pdeDomain,model_fcn,params_pde,params_solver,device)
optimizer = torch.optim.Adam(mypde.model.parameters(), lr=2.e-4,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

# resampler = Resampler(mypde,pdeData,save_dir=save_dir,interval=6e4,min_iter=4e4,max_iter=4e5,method='EVO')
# IRDR_attention = IRDR_BRDRLayer(mode=all)

train(mypde,pdeData,**params_train,attention_loss=None)

# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))
solution = mypde.full_predict(pdeData.X_test)
phi_pred= mypde.unpack_solution(solution)[0] 
ngroup =2
mesh = CartesianGeometry(ndim=2,xmin=params_domain['xmin'],xmax=params_domain['xmax'],num_cells=args.n_test)
visualization.export_flux_list(mesh,phi_pred,filename= save_dir+'flux_data.vtr')
visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,n_phi_groups=ngroup,n_p_groups=ngroup)

        
##==========================================================================
for igroup in range(ngroup):
    visualization.viewFlux(mesh,phi_pred[igroup],save_dir,'flux_group_'+str(igroup)+'.png',index='xy')

show_flux = True
if show_flux:
    phi_AE, phi_pred, phi_test,_, _ = mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)[2:]
    for igroup in range(ngroup):
        visualization.viewErrorAndFlux(mesh,phi_AE[igroup],phi_pred[igroup],phi_test[igroup],
                                        save_dir,'error_phi_group_'+str(igroup)+'.png',index='xy')

show_currents = True
if show_currents:
    p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=False)[2:]
    for igroup in range(ngroup):
        visualization.viewErrorAndCurrents(mesh,p_AE[igroup],p_pred[igroup],p_test[igroup],
                                            save_dir,'error_current_group_'+str(igroup)+'.png',index='xy')
        

