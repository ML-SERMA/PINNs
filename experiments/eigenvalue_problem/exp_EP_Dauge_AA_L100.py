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
parser.add_argument('-t','--test', type=str, help="name of test case",default='EP_Mixed_FCN_Dauge_L100_D5_MSLR1e-3_NIT5000_HBC_AA_beta0.8_m10')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=10*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test',nargs='+', type=int, help="an integer number",default=[120,120])
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=2000000)
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



params_domain = {'xmin':[0, 0], 'xmax':[100, 100]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,
                'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,'resampling_every':0,
                'normalization':True}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':'mixed_L100'}

params_solver = {'momentum':False,'beta1':0.0,'beta2':0.4,
                'num_inner_iters':5000, 'keff_ref':0.99513,'verbose':2,
                'anderson':True,'beta':0.8,'m':10}


def func_D(X):
    Dl = 1.0
    Dh = 5.0

    x=X[:,0:1]
    y=X[:,1:2]
    D = Dl*torch.ones_like(x)
    D[(x>=50)&(y<=50)] = Dh
    D[(x<=50)&(y>=50)] = Dh
    return D

def func_Sigma_a(X):
    return torch.ones_like(X[:,0:1])

def func_Sigma_f(X):
    return torch.ones_like(X[:,0:1])





params_pde = {'func_D':func_D,'func_Sigma_a':func_Sigma_a,'func_Sigma_f':func_Sigma_f}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_phi_test(load_dir = data_dir+'EP_Dauge_L100_D5/RefFlux120')
pdeData.generate_p_test(load_dir = data_dir+'EP_Dauge_L100_D5/RefCurrents120')
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
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000,200000,300000,400000,500000,600000], gamma=0.2)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)
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
visualization.viewErrorAndSolution(params_domain,[120,120],phi_AE,phi_pred,phi_test,save_dir)
visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_phi=True,Error_p=True)

# Visualization currents
show_currents = True
if show_currents:
    p_REL_L2 ,p_AE, p_pred, p_test = mypde.get_currents_test(pdeData.Xc_test,pdeData.p_test,normalization=True,mass_normalization=mass_phi)
    visualization.viewErrorAndCurrents(params_domain,[120,120],p_AE,p_pred,p_test,save_dir)
    # visualization.viewCrossSection(pdeData.Xc_test[0],S=p_pred[0],pathFile=save_dir+'p_pred0.png')
    # visualization.viewCrossSection(pdeData.Xc_test[1],S=p_pred[1],pathFile=save_dir+'p_pred1.png')


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


