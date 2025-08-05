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
from pyPINNs.Xs.cross_sections import XsContainer,XsEvaluator
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.resampler import Resampler
# from pyPINNs.Tools.adaptive_loss import ResidualAttention, ResidualAttentionMLP
from pyPINNs.Tools.attention_loss import ResidualWeightingAttention
from pyPINNs.Tools.saveResult import saveResult
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
parser.add_argument('-t','--test', type=str, help="name of test case",default='EP_Mixed_FCN_2G_TWIGL_2D_SLR1e-3_09_Nit2000_HBC')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=20*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[120,120])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=2000000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[2]+5*[64]+[6])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Sin')
parser.add_argument('-s','--sampling', type=str, help="sampling method",default='Sobol')
parser.add_argument('-v', '--verbose',action='count', default=0)  

args = parser.parse_args()

name_folder = args.test + '_nc' + str(args.n_collocation) + '_nb' + str(args.n_boundary) \
                + '_nt' + str(args.n_test) + '_ns' + str(args.n_step) + '_nn' + str(args.n_neuron)+'_'+str(args.activation)+'_'+str(args.sampling)
save_dir = check_create_dir(results_dir+name_folder+'/')
print(f"==>> save_dir: {name_folder}")

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[80, 80],'boundary_conditions': [['REFLECTION', 'ZERO_FLUX'],['REFLECTION', 'ZERO_FLUX']],
                'type_of_points':args.sampling}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'num_groups':2,'normalization':True}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'TWIGL_2D'}

params_solver = {'momentum':False,'beta1':0.0,'beta2':0.6,
                'num_inner_iters':2000, 'keff_ref':0.91307,'verbose':2, 'save_dir':save_dir,
                'anderson':False,'beta':0.8,'m':4}



pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'TWIGL_2D/RefFlux120',multigroup=True)
pdeData.generate_p_test(load_dir = data_dir+'TWIGL_2D/RefCurrents120',multigroup=True)




geometry = CartesianGeometry(ndim=2,xmin=[0,0],xmax=[80, 80],num_cells=[3,3],device=device)
geometry = geometry.from_edges(edges=[[0,24,56,80],[0,24,56,80]],device=device)
# geometry.view_grid()

materialName = [ "M1" , "M2" ,"M3"]
nmat = len(materialName)
M1 = 0
M2 = 1
M3 = 2
regionsName=["M1", "M2","M3"]
regions =  [[M3, 0, 3, 0, 3],
            [M1, 1, 2 , 1, 2],
            [M2, 1, 2 , 0, 1],
            [M2, 0, 1 , 1, 2],
            ]

geometry.setRegionsName(regionsName=regionsName)
geometry.setRegionsIndex(regionsIndex=regions)

material_map = geometry.construct_material_map()
print(f"==>> material_map: {material_map}")

ngroup =2
xsContainer = XsContainer(num_groups=ngroup,num_materials=nmat,device=device)
tD0 = [1.4, 1.4, 1.3  ]
tD1 = [0.4, 0.4, 0.5  ]
xsContainer.setDiffusion(tD0, group_idx=0)
xsContainer.setDiffusion(tD1, group_idx=1)



tUNS3D0 = [ 1./(3*1.4), 1./(3*1.4), 1./(3*1.3)]
tUNS3D1 = [ 1./(3*0.4), 1./(3*0.4), 1./(3*0.5)]
xsContainer.setUNS3D(tUNS3D0, 0)
xsContainer.setUNS3D(tUNS3D1, 1)


tRemoval0 = [0.01+0.01, 0.01+0.01, 0.008+0.01]
tRemoval1 = [0.15 , 0.15, 0.05]
xsContainer.setRemoval(tRemoval0, 0)
xsContainer.setRemoval(tRemoval1, 1)


tScatter00 = [ 0., 0., 0.]
tScatter01 = [ 0.,  0. , 0.]
tScatter10 = [ 0.01,  0.01, 0.01]
tScatter11 = [ 0.,  0. , 0.]

xsContainer.setScattering(tScatter00, 0, 0)
xsContainer.setScattering(tScatter10, 1, 0)
xsContainer.setScattering(tScatter01, 0, 1)
xsContainer.setScattering(tScatter11, 1, 1)


tNusigf0 = [ 0.007, 0.007, 0.003]
tNusigf1 = [ 0.2  , 0.2  , 0.06  ]
xsContainer.setNuSigf(tNusigf0, 0)
xsContainer.setNuSigf(tNusigf1, 1)

tSpectrum = [1.0, 0]
xsContainer.setFissionSpectrum(tSpectrum)

xsEvaluater = XsEvaluator(xsContainer,geometry)

# D = xsEvaluater.diffusion(pdeData.X_train)
# Sigma_r = xsEvaluater.sigma_r(pdeData.X_train)
# Sigma_s = xsEvaluater.sigma_s(pdeData.X_train)
# NuSigma_f = xsEvaluater.nusigma_f(pdeData.X_train)
# spectrum = xsEvaluater.fission_spectrum()


params_pde = {'XsEvaluater': xsEvaluater, 'num_groups':2}



# visualization.viewCrossSection(pdeData.X_train,func_D1(pdeData.X_train),pathFile=save_dir+'D1.png')
# visualization.viewCrossSection(pdeData.X_train,func_D2(pdeData.X_train),pathFile=save_dir+'D2.png')
# visualization.viewCrossSection(pdeData.X_train,func_Sigma_a1(pdeData.X_train),pathFile=save_dir+'Sigma_a1.png')
# visualization.viewCrossSection(pdeData.X_train,func_Sigma_a2(pdeData.X_train),pathFile=save_dir+'Sigma_a2.png')
# visualization.viewCrossSection(pdeData.X_train,func_Sigma_s12(pdeData.X_train),pathFile=save_dir+'Sigma_s12.png')
# visualization.viewCrossSection(pdeData.X_train,func_Sigma_f1(pdeData.X_train),pathFile=save_dir+'Sigma_f1.png')
# visualization.viewCrossSection(pdeData.X_train,func_Sigma_f2(pdeData.X_train),pathFile=save_dir+'Sigma_f2.png')


model_fcn = FCN_FF(**params_model).to(device)
summary(model_fcn, input_size=(2,))
print('model',model_fcn)



# Training Time
mypde = mixed_multigroup_diffusion_eigenvalue(pdeDomain,model_fcn,params_pde,params_solver,device)
optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

# esampler = Resampler(mypde,pdeData,save_dir=save_dir,interval=1e5,min_iter=1e5,max_iter=1e6,method='EVO')
# RA_loss = ResidualAttention(lr=1e-2, temperature=1.0, mode="all")

#RAN_loss = ResidualAttentionMLP(hidden_dim=32,lr=0.01).to(device)

# load model before train
# mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))

AttentionLoss = ResidualWeightingAttention(shape=(args.n_collocation,2,3),learn_rate=0.2, init="ones", normalize=None,device=device)

train(mypde,pdeData,**params_train,attention_loss=None)


# # continue to train with LBFGS
# params_train['n_step']= 4000
# mypde.N_inner=100
# optimizer_LBFGS = torch.optim.LBFGS(mypde.model.parameters(),lr=0.1,line_search_fn="strong_wolfe")
# mypde.compile(optimizer=optimizer_LBFGS,scheduler=None)
# train(mypde,pdeData,**params_train,mode='a')






# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))
solution = mypde.full_predict(pdeData.X_test)

phi_pred= mypde.unpack_solution(solution)[0] 

mesh = CartesianGeometry(ndim=2,xmin=params_domain['xmin'],xmax=params_domain['xmax'],num_cells=args.n_test)
visualization.export_flux_list(mesh,phi_pred,filename= save_dir+'flux_data.vtr')

visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=True,Error_p=True)

for igroup in range(ngroup):
    visualization.viewFlux(mesh,phi_pred[igroup],save_dir,'flux_group_'+str(igroup)+'.png',order='C')



show_flux = True
if show_flux:
    phi_REL_L2, phi_AE, phi_pred, phi_test,mass_phi_pred, mass_phi_test = mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=True)
    for igroup in range(ngroup):
        visualization.viewErrorAndFlux(mesh,phi_AE[igroup],phi_pred[igroup],phi_test[igroup],
                                        save_dir,'error_phi_group_'+str(igroup)+'.png',order='C')

show_currents = True
if show_currents:
    p_REL_L2 ,p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=True,
                                                                mass_pred=mass_phi_pred,mass_test=mass_phi_test)
    for igroup in range(ngroup):
        visualization.viewErrorAndCurrents(mesh,p_AE[igroup],p_pred[igroup],p_test[igroup],
                                            save_dir,'error_current_group_'+str(igroup)+'.png',order='C')
        
visualization.show_history_residuals(pathFile=save_dir+'train.csv',save_dir=save_dir,keff_ref_exist=True)


























