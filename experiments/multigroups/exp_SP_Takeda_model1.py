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
from pyPINNs.PDE.mixed_multigroup_diffusion_source import mixed_multigroup_diffusion_source
from pyPINNs.Geometry.CartesianGeometry import CartesianGeometry
from pyPINNs.Mesh.CartesianMesh import CartesianMesh
from pyPINNs.Xs.cross_sections import XsContainer,XsEvaluator
from pyPINNs.Tools.visualization import visualization
from pyPINNs.Tools.resampler import Resampler
from pyPINNs.Tools.adaptive_loss import ResidualAttention
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
parser.add_argument('-t','--test', type=str, help="name of test case",default='newSP_Mixed_FCN_Takeda_model1_SLR1e-3')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=100*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="list of integer number",default=[50,50,50])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=500000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="list of integer number",default=[3]+5*[64]+[8])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Sin')
parser.add_argument('-s','--sampling', type=str, help="sampling method",default='Sobol')
parser.add_argument('-v', '--verbose',action='count', default=0)  

args = parser.parse_args()

name_folder = args.test + '_nc' + str(args.n_collocation) + '_nb' + str(args.n_boundary) \
                + '_nt' + str(args.n_test) + '_ns' + str(args.n_step) + '_nn' + str(args.n_neuron)\
                +'_'+str(args.activation) +'_'+str(args.sampling)
save_dir = check_create_dir(results_dir+name_folder+'/')
print(f"==>> save_dir: {name_folder}")

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(f"Running on {device}. Yes! ")

# device = torch.device('cpu')

params_domain = {'xmin':[0, 0, 0], 'xmax':[25, 25, 25],
                'boundary_conditions': [['REFLECTION', 'VACUUM'],['REFLECTION', 'VACUUM'],['REFLECTION', 'VACUUM']],
                'type_of_points':args.sampling}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'num_groups':2,'normalization':False} # we do not use normalization for the source problem

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'Takeda1'}

# params_solver = {'momentum':False,'beta1':0.0,'beta2':0.8,
#                 'num_inner_iters':1000, 'keff_ref':0.92818,'verbose':2, 'save_dir':save_dir, 'keff_method':'scaled_rayleigh',
#                 'anderson':False,'beta':0.8,'m':4} 






pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'TakedaModel1Source/RefFlux50',multigroup=True)
pdeData.generate_p_test(load_dir = data_dir+'TakedaModel1Source/RefCurrents50',multigroup=True)




geometry = CartesianGeometry(ndim=3,xmin=[0, 0, 0],xmax=[25, 25, 25],num_cells=[5, 5, 5],device=device)
materialName = [ "COR" , "REF" ,"VOID"]
nmat = len(materialName)
COR = 0
REF = 1
VOID = 2
regionsName=[ "COR" , "REF" ,"VOID"]
regions =  [[REF, 0, 5, 0, 5, 0, 5],
            [COR, 0, 3, 0, 3, 0, 3],
            [VOID,  3, 4, 0, 1, 0, 5],
            ]

geometry.setRegionsName(regionsName=regionsName)
geometry.setRegionsIndex(regionsIndex=regions)

material_map = geometry.construct_material_map()
print(f"==>> material_map: {material_map}")

# geometry.plot_material_map_with_indices(material_map)
# geometry.plot_material_voxels(material_map)


ngroup =2
xsContainer = XsContainer(num_groups=ngroup,num_materials=nmat,device=device)


tTotal0 = [ 2.23775E-01 , 2.50367E-01 , 1.28407E-02 ]
tTotal1 = [ 1.03864 , 1.64482 ,  1.20676E-02  ]
xsContainer.setTotal(tTotal0, 0)
xsContainer.setTotal(tTotal1, 1)




tD0 = [1/(3*tTotal) for tTotal in tTotal0]
tD1 = [1/(3*tTotal) for tTotal in tTotal1]

xsContainer.setDiffusion(tD0, group_idx=0)
xsContainer.setDiffusion(tD1, group_idx=1)



# tUNS3D0 = [ 1./(3*1.2), 1./(3*1.2), 1./(3*1.2)]
# tUNS3D1 = [ 1./(3*0.4), 1./(3*0.4), 1./(3*0.2)]
# xsContainer.setUNS3D(tUNS3D0, 0)
# xsContainer.setUNS3D(tUNS3D1, 1)





tScatter00 = [  1.92423E-01, 1.93446E-01, 1.27700E-02 ]
tScatter01 = [ 0.0,0.0,0.0] 
tScatter10 = [  2.28253E-02 , 5.65042E-02,  2.40997E-05 ]
tScatter11 = [ 8.80439E-01 , 1.62452 , 1.07387E-02 ]

xsContainer.setScattering(tScatter00, 0, 0)
xsContainer.setScattering(tScatter10, 1, 0)
xsContainer.setScattering(tScatter01, 0, 1)
xsContainer.setScattering(tScatter11, 1, 1)


tRemoval0 = [tTotal0[i]-tScatter00[i] for i in range(len(tTotal0))]
tRemoval1 = [tTotal1[i]-tScatter11[i] for i in range(len(tTotal1))]
xsContainer.setRemoval(tRemoval0, 0)
xsContainer.setRemoval(tRemoval1, 1)



# tNusigf0 = [ 9.09319E-03, 0.0 , 0.0 ]
# tNusigf1 = [ 2.90183E-01, 0.0 , 0.0 ]
# xsContainer.setNuSigf(tNusigf0, 0)
# xsContainer.setNuSigf(tNusigf1, 1)

# tSpectrum = [1.0, 0]
# xsContainer.setFissionSpectrum(tSpectrum)

tSf0 = [ 9.09319E-03, 0.0 , 0.0 ]
tSf1 = [ 2.90183E-01, 0.0 , 0.0 ]
xsContainer.setSource(tSf0, 0)
xsContainer.setSource(tSf1, 1)






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
summary(model_fcn, input_size=(3,))
print('model',model_fcn)



# Training Time
# mypde = mixed_multigroup_diffusion_eigenvalue(pdeDomain,model_fcn,params_pde,params_solver,device)
mypde = mixed_multigroup_diffusion_source(pdeDomain,model_fcn,params_pde,device)

#Adam
optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.95)
mypde.compile(optimizer=optimizer,scheduler=scheduler)


# LBFGS
# optimizer_LBFGS = torch.optim.LBFGS(mypde.model.parameters(),lr=0.1,line_search_fn="strong_wolfe")
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_LBFGS, step_size=200, gamma=0.95)
# mypde.compile(optimizer=optimizer_LBFGS,scheduler=scheduler)


# resampler = Resampler(mypde,pdeData,save_dir=save_dir,interval=1e5,min_iter=1e5,max_iter=1e6,method='EVO')
# RA_loss = ResidualAttention(lr=1e-3, temperature=1.0, mode="all")

# load model before train
# mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))

train(mypde,pdeData,**params_train,adaptive_loss=None,resampler=None)



# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))
solution = mypde.full_predict(pdeData.X_test)

phi_pred= mypde.unpack_solution(solution)[0] 

mesh = CartesianGeometry(ndim=3,xmin=params_domain['xmin'],xmax=params_domain['xmax'],num_cells=args.n_test)
visualization.export_flux_list(mesh,phi_pred,filename= save_dir+'flux_data.vtr')


visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=True,Error_p=True)

ngroup =2
for igroup in range(ngroup):
    visualization.viewFlux(mesh,phi_pred[igroup],save_dir,'flux_group_'+str(igroup)+'.png',order='F')


show_flux = True
if show_flux:
    phi_REL_L2, phi_AE, phi_pred, phi_test,mass_phi_pred, mass_phi_test = mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=False)
    for igroup in range(ngroup):
        visualization.viewErrorAndFlux(mesh,phi_AE[igroup],phi_pred[igroup],phi_test[igroup],
                                        save_dir,'error_phi_group_'+str(igroup)+'.png',order='F')

show_currents = True
if show_currents:
    p_REL_L2 ,p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=False,
                                                                mass_pred=mass_phi_pred,mass_test=mass_phi_test)
    for igroup in range(ngroup):
        visualization.viewErrorAndCurrents(mesh,p_AE[igroup],p_pred[igroup],p_test[igroup],
                                            save_dir,'error_current_group_'+str(igroup)+'.png',order='F')
        
























