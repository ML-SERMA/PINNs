import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/'+'EP_mixed_Center_NIT_type2/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='EP_Center_L100_D5')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_p')
args = parser.parse_args()


test_case= args.test 
Error_phi = True
type_loss = args.loss

save_dir = 'Compare_NIT_HBC_'+test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")



 

Folders = ['EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT1000_HBC_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
           'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
           'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT3000_HBC_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
           'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT4000_HBC_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
        ]
legend=[r'$N=1000$',r'$N=2000$',r'$N=3000$',r'$N=4000$'] 








list_data = []

for Folder in Folders:
    pathFile = results_dir+Folder+'/Loss.csv'
    if Error_phi:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi','Error_p'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    list_data.append(df)




ext = '.pdf'
fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True)
plt.legend(legend,ncol=2)
plt.xlabel('Iteration')
plt.ylim([1.e-2,10])
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' + type_loss +ext)
plt.show()


## residual 


list_ResFlux = []
list_ResKeff = []
list_AccKeff = []

for Folder in Folders:
    df_ResFlux = pd.read_csv( results_dir+Folder+'/ResFlux.csv')
    df_ResKeff = pd.read_csv( results_dir+Folder+'/ResKeff.csv')
    df_AccKeff = pd.read_csv( results_dir+Folder+'/AccKeff.csv')

    list_ResFlux.append(df_ResFlux)
    list_ResKeff.append(df_ResKeff)
    list_AccKeff.append(df_AccKeff)

fig, ax = plt.subplots()
for df in list_ResFlux:
    df.plot(x='Iteration',y='Value',ax=ax,logy=True)
plt.legend(legend,ncol=2)
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' +'ResFlux'+ext)
plt.show()

fig, ax = plt.subplots()
for df in list_ResKeff:
    df.plot(x='Iteration',y='Value',ax=ax,logy=True)
plt.legend(legend,ncol=2)
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' +'ResKeff'+ext)
plt.show()

fig, ax = plt.subplots()
for df in list_AccKeff:
    df.plot(x='Iteration',y='Value',ax=ax,logy=True)
plt.legend(legend,ncol=2)
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' +'AccKeff'+ext)
plt.show()
    