import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/'+'EP_mixed_Dauge_AA_type2_NIT/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='EP_Dauge_L100_D5')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_phi')
args = parser.parse_args()


test_case= args.test 
Error_phi = True
type_loss = args.loss

save_dir = 'AA_HBC__beta0.8_m10_'+test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")



# Folders = ['EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
#         'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_AA_beta0.2_m4_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
#         'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_AA_beta0.4_m4_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
#         'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_AA_beta0.6_m4_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
#         'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_AA_beta0.8_m4_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
#         'EP_Mixed_FCN_Center_L100_D5_MSLR1e-3_NIT2000_HBC_AA_beta0.9_m4_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
#         ]
# legend=[r'Without AA',r'AA-$\rho=0.2$',r'AA-$\rho=0.4$',r'AA-$\rho=0.6$',r'AA-$\rho=0.8$',r'AA-$\rho=0.9$'] 




            # 
Folders = [ #'EP_Mixed_FCN_Dauge_L100_D5_MSLR1e-3_NIT1000_HBC_AA_beta0.8_m10_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'EP_Mixed_FCN_Dauge_L100_D5_MSLR1e-3_NIT2000_HBC_AA_beta0.8_m10_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'EP_Mixed_FCN_Dauge_L100_D5_MSLR1e-3_NIT3000_HBC_AA_beta0.8_m10_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'EP_Mixed_FCN_Dauge_L100_D5_MSLR1e-3_NIT4000_HBC_AA_beta0.8_m10_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'EP_Mixed_FCN_Dauge_L100_D5_MSLR1e-3_NIT5000_HBC_AA_beta0.8_m10_nc10240_nb512_nt[120, 120]_ns2000000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
        ]
legend=[r'AA-$N=2000$',r'AA-$N=3000$',r'AA-$N=4000$',r'AA-$N=5000$'] 






list_data = []

for Folder in Folders:
    pathFile = results_dir+Folder+'/Loss.csv'
    print(f"==>> pathFile: {pathFile}")
    if Error_phi:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi','Error_p'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    list_data.append(df)




ext = '.pdf'
fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True)
plt.legend(legend,ncol=1)
plt.xlabel('Iteration')
# plt.ylim([1.e-2,5])
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
plt.legend(legend,ncol=1)
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' +'ResFlux'+ext)
plt.show()

fig, ax = plt.subplots()
for df in list_ResKeff:
    df.plot(x='Iteration',y='Value',ax=ax,logy=True)
plt.legend(legend,ncol=1)
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' +'ResKeff'+ext)
plt.show()

fig, ax = plt.subplots()
for df in list_AccKeff:
    df.plot(x='Iteration',y='Value',ax=ax,logy=True)
plt.legend(legend,ncol=1)
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' +'AccKeff'+ext)
plt.show()
    