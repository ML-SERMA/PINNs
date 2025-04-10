import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/Mixed_SP/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_Dauge_L1')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_p')
args = parser.parse_args()


test_case= args.test 
mixed = True
Center = False
Dauge  = True
type_loss = args.loss

save_dir = test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")

if Dauge:
    print('Dauge test')
    Folders = [ 'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Dauge_D2_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Dauge_D2_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Dauge_D2_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Dauge_D10_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Dauge_D10_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Dauge_D10_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            ]
if Center:
    print('Center test')
    Folders = [ 'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Center_D2_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Center_D2_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Center_D2_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Center_D10_SLR1e-3_2000_095_SBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Center_D10_SLR1e-3_2000_095_LRA_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
                'Mixed_FCN_Center_D10_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            ]
 
# legend=['D1_SBC','D1_LRA','D1_HBC','D2_SBC','D2_LRA','D2_HBC','D10_SBC','D10_LRA','D10_HBC']  
legend=[r'$\mathcal{D}=1$-SBC',r'$\mathcal{D}=1$-LRA',r'$\mathcal{D}=1$-HBC',
        r'$\mathcal{D}=2$-SBC',r'$\mathcal{D}=2$-LRA',r'$\mathcal{D}=2$-HBC',
        r'$\mathcal{D}=10$-SBC',r'$\mathcal{D}=10$-LRA',r'$\mathcal{D}=10$-HBC'
        ]  


list_data = []

for Folder in Folders:
    pathFile = results_dir+Folder+'/Loss.csv'
    if mixed:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi','Error_p'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi'])
    list_data.append(df)


ext ='.pdf'

fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True)
plt.legend(legend,ncol=3)
plt.xlabel('Iteration')
fig.tight_layout()
plt.ylim([5.e-3,5])
plt.savefig(save_dir+ test_case + '_' + type_loss +ext)
# plt.show()