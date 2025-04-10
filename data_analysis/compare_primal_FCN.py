import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/Primal_SP/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='Primal_Center_L1')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Loss')
args = parser.parse_args()


test_case= args.test 
Error_phi = True
type_loss = args.loss


save_dir = test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")

Folders = [ 'Primal_FCN_Constant_D1_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Constant_D1_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Constant_D1_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Center_D2_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Center_D2_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Center_D2_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Center_D10_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Center_D10_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
            'Primal_FCN_Center_D10_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
        ]
 
# Folders = [ 'Primal_FCN_Constant_D1_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Constant_D1_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Constant_D1_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Dauge_D2_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Dauge_D2_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Dauge_D2_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Dauge_D10_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Dauge_D10_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#             'Primal_FCN_Dauge_D10_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 1]_Tanh',
#         ]
 


# legend=['D1_SBC','D1_LRA','D1_HBC','D2_SBC','D2_LRA','D2_HBC','D10_SBC','D10_LRA','D10_HBC']  
legend=[r'$\mathcal{D}=1$-SBC',r'$\mathcal{D}=1$-LRA',r'$\mathcal{D}=1$-HBC',
        r'$\mathcal{D}=2$-SBC',r'$\mathcal{D}=2$-LRA',r'$\mathcal{D}=2$-HBC',
        r'$\mathcal{D}=10$-SBC',r'$\mathcal{D}=10$-LRA',r'$\mathcal{D}=10$-HBC'
        ]  



list_data = []

for Folder in Folders:
    pathFile = results_dir+Folder+'/Loss.csv'
    if Error_phi:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    list_data.append(df)




ext = '.pdf' 
fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True)
plt.legend(legend,ncol=3)
# plt.ylim([1.e-8,12])
plt.xlabel('Iteration')
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' + type_loss +ext)

plt.show()