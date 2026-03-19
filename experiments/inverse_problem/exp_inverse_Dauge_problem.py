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
from pyPINNs.PDE_dev.primal_one_group_diffusion_source import primal_one_group_diffusion_source

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
parser.add_argument('-t','--test', type=str, help="name of test case",default='Inverse_FCN_Dauge_D10_MSLR1e-3_HBC')
parser.add_argument('-nc','--n_collocation', type=int, help="an integer number",default=40*1024)
parser.add_argument('-nb','--n_boundary', type=int, help="an integer number",default=512)
parser.add_argument('-nt','--n_test', type=int, help="an integer number",default=120)
parser.add_argument('-ns','--n_step', type=int, help="an integer number",default=200000)
parser.add_argument('-log','--log_every', type=int, help="an integer number",default=100)
parser.add_argument('-nn','--n_neuron',nargs='+', type=int, help="an integer number",default=[2]+6*[64]+[1])
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



params_domain = {'xmin':[0, 0], 'xmax':[1, 1]}
params_data = {'n_collocation':args.n_collocation,'n_boundary':args.n_boundary,'n_test':args.n_test,
                'random_seed':2024,'device':device}

params_train = {'n_step':args.n_step,'verbose':args.verbose,'log_every':args.log_every,'LossFile':'Loss.csv','save_dir':save_dir,
                'train_on_batch': False, 'learning_rate_annealing':False ,'weight_ridge':0,'weight_lasso':0,'weight_Sobolev':0,'resampling_every':0}

params_model = {'layers':args.n_neuron,'activation':args.activation,'device':device,
                'fourier_mapping_size':None,'hard_BC':True,'transform':False}

pdeDomain = SquareDomain(**params_domain)
pdeData = DataSet(domain=pdeDomain,**params_data)
pdeData.generate_u_test(load_dir = data_dir+'Dauge_D10/')
# pdeData.prepare_DataLoader(batch_size_collocation=None,batch_size_boundary=None)


class func_D(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.Dh = torch.nn.Parameter(torch.tensor(1.0).float().to(device))
        self.Dl = torch.nn.Parameter(torch.tensor(1.0).float().to(device))
    def forward(self, X):
        x=X[:,0:1]
        y=X[:,1:2]
        D = self.Dl*torch.ones_like(x)
        D[(x>=0.5)&(y<=0.5)] = self.Dh
        D[(x<=0.5)&(y>=0.5)] = self.Dh
        return D

def func_Sigma_a(X):
    return torch.ones_like(X[:,0:1])

def func_Source(X):
    return torch.ones_like(X[:,0:1])

# model_D   = FCN(layers=[2,16,16,1],activation='Tanh',device=device,hard_BC =False,transform=False).to(device)
model_D = func_D(device=device).to(device)

params_pde = {'func_D':model_D,'func_Sigma_a':func_Sigma_a,'func_Source':func_Source}


model_fcn = FCN_FF(**params_model).to(device)
# Training Time
mypde = primal_one_group_diffusion_source(pdeDomain,model_fcn,params_pde,device)

summary(model_fcn, input_size=(2,))
print('model',model_fcn)

input('Enter')

parameters = [*mypde.model.parameters(), *model_D.parameters()]

optimizer = torch.optim.Adam(parameters, lr=1.e-3,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000,40000,60000,80000,100000], gamma=0.5)
mypde.compile(optimizer=optimizer,scheduler=scheduler)

train(mypde,pdeData,**params_train)
# mypde.fit(pdeData,**params_train)

# Model Accuracy 
mypde.model.load_state_dict(torch.load(save_dir+"model.pt"))

error_relative, error_absolute, u_pred, u_test = mypde.get_accuracy(pdeData.X_test,pdeData.u_test)
error_relative = error_relative.cpu().detach().numpy()
error_absolute = error_absolute.cpu().detach().numpy()
u_pred = u_pred.cpu().detach().numpy()
u_test = u_test.cpu().detach().numpy()

# Visualization
visualization.viewErrorAndSolution(params_domain,120,error_absolute,u_pred,u_test,save_dir)
visualization.show_loss(pathFile=save_dir+'Loss.csv',save_dir=save_dir,Error_Test=True)

D = model_D(pdeData.X_test)
D = D.cpu().detach().numpy()
visualization.viewSolution(params_domain,120,D,save_dir)

