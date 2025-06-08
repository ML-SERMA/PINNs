import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='Mixed_Constant_L1')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_phi')
args = parser.parse_args()


test_case= args.test 
mixed = True
Center = False
Dauge  = True
type_loss = args.loss

save_dir = test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")


Folders = [
            'Mixed_FCN_Constant_D1_SLR1e-3_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'Mixed_FCN_Constant_D1_MSLR1e-3_2000_095_HBC_nc20480_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'Mixed_FCN_Constant_D1_MSLR1e-4_2000_095_HBC_nc10240_nb512_nt[120, 120]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh'
        ]

legend=[r'$N_r=10240$-0.001', r'$N_r=20480$-0.001', r'$N_r=10240$-0.0001']  


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
# plt.legend(legend,ncol=3)
plt.xlabel('Iteration')
fig.tight_layout()
# plt.ylim([5.e-3,5])
plt.savefig(save_dir+ test_case + '_' + type_loss +ext)
# plt.show()