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
from pyPINNs.PDE.mixed_eigenvalue_one_group import mixed_eigenvalue_one_group
from pyPINNs.Mesh.CartesianMesh import CartesianMesh
from pyPINNs.Tools.visualization import visualization
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
parser.add_argument('-t','--test', type=str, help="name of test case",default='EP_Mixed_FCN_LWR_MSLR1e-3_NIT5000_HBC_MA_beta0.8')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=200*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="an integer number",default=[96,86])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=1000000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="an integer number",default=[2]+5*[64]+[3])
parser.add_argument('-a','--activation', type=str, help="activation function",default='Tanh')
parser.add_argument('-v', '--verbose',action='count', default=0)  

args = parser.parse_args()

name_folder = args.test + '_nc' + str(args.n_collocation) + '_nb' + str(args.n_boundary) \
                + '_nt' + str(args.n_test)  + '_ns' + str(args.n_step) + '_nn' + str(args.n_neuron)+'_'+str(args.activation)
save_dir = check_create_dir(results_dir+name_folder+'/')

# Check if we run on GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    # torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
    
# device = torch.device('cpu')
print(f"Running on {device}. Yes! ")



params_domain = {'xmin':[0, 0], 'xmax':[96, 86],'boundary_conditions': [['VACUUM', 'VACUUM'],['VACUUM', 'VACUUM']]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,
                'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,'resampling_every':0,
                'normalization':True}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'mixed_vacuum_96_86'}

params_solver = {'momentum':True,'beta1':0.0,'beta2':0.8,
                'num_inner_iters':5000, 'keff_ref':1.006930,'verbose':2,
                'anderson':False,'beta':0.6,'m':4}


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



def func_Sigma_f(X):
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





params_pde = {'func_D':func_D,'func_Sigma_a':func_Sigma_a,'func_Sigma_f':func_Sigma_f}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'EP_IAEA-5S/RefFlux_96_86')
pdeData.generate_p_test(load_dir = data_dir+'EP_IAEA-5S/RefCurrents_96_86')
# pdeData.prepare_DataLoader(batch_size_collocation=None,batch_size_boundary=None)





# visualization.viewCrossSection(pdeData.Xc_test[0],S=pdeData.p_test[0],pathFile=save_dir+'Xc0.png')
# visualization.viewCrossSection(pdeData.Xc_test[1],S=pdeData.p_test[1],pathFile=save_dir+'Xc1.png')
# visualization.viewCrossSection(pdeData.X_train,func_D(pdeData.X_train))
# visualization.viewCrossSection(pdeData.X_test,func_D(pdeData.X_test))


model_fcn = FCN_FF(**params_model).to(device)

summary(model_fcn, input_size=(2,))
print('model',model_fcn)

input('Enter')

# Training Time
mypde = mixed_eigenvalue_one_group(pdeDomain,model_fcn,params_pde,params_solver,device)

optimizer = torch.optim.Adam(mypde.model.parameters(), lr=1.e-3,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000,100000,200000,300000,400000,500000], gamma=0.2)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.95)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

#====== Train =====================
train(mypde,pdeData,**params_train)


# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))
phi_REL_L2, phi_AE, phi_pred, phi_test, mass_phi = mypde.get_phi_test(pdeData.X_test,pdeData.phi_test,normalization=True)
phi_REL_L2 = phi_REL_L2.cpu().detach().numpy()
phi_AE = phi_AE.cpu().detach().numpy()
phi_pred = phi_pred.cpu().detach().numpy()
phi_test = phi_test.cpu().detach().numpy()

# Visualization Flux
visualization.viewErrorAndSolution(params_domain,[96,86],phi_AE,phi_pred,phi_test,save_dir)
visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=True,Error_p=True)

# Visualization currents
show_currents = True
if show_currents:
    p_REL_L2 ,p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=True,mass_normalization=mass_phi)
    visualization.viewErrorAndCurrents(params_domain,[96,86],p_AE,p_pred,p_test,save_dir)
    


# Save residual
## Save data
saveResult.saveData(mypde.resKeff,save_dir,'ResKeff')
saveResult.saveData(mypde.resFlux,save_dir,'ResFlux')
saveResult.saveData(mypde.evoKeff,save_dir,'EvoKeff')
saveResult.saveData(mypde.accKeff,save_dir,'AccKeff')

visualization.viewEvolutionResidual(mypde.resKeff,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'ResKeff','color':'blue'})
saveResult.saveFigures(save_dir,'ResKeff')
visualization.viewEvolutionResidual(mypde.resFlux,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'ResFlux','color':'green'})
saveResult.saveFigures(save_dir,'ResFlux')

visualization.viewEvolutionResidual(mypde.evoKeff,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'EvoKeff','color':'blue'})
saveResult.saveFigures(save_dir,'EvoKeff')
visualization.viewEvolutionResidual(mypde.accKeff,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'AccKeff','color':'green'})
saveResult.saveFigures(save_dir,'AccKeff')



saveResult.saveData(mypde.evoNit,save_dir,'EvoNit')
saveResult.saveData(mypde.evoInnerSolver,save_dir,'evoInnerSolver')
saveResult.saveData(mypde.resIS,save_dir,'resIS')
visualization.viewEvolutionResidual(mypde.evoNit,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'EvoNit','color':'blue'})
saveResult.saveFigures(save_dir,'EvoNit')
visualization.viewEvolutionResidual(mypde.evoInnerSolver,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'evoInnerSolver','color':'red'})
saveResult.saveFigures(save_dir,'evoInnerSolver')
visualization.viewEvolutionResidual(mypde.resIS,paramFigure={'title':'','xlabel':'Iteration','ylabel':'','label':'resIS','color':'red'})
saveResult.saveFigures(save_dir,'resIS')

