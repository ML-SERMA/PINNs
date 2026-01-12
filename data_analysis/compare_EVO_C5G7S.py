import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/'+'Compare_EVO/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='C5G7S')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_p')
args = parser.parse_args()


test_case= args.test 
Error_phi = True
type_loss = args.loss

save_dir = 'SP_'+test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")




Folders = [ 'Scaled_Mixed_FCN_2G_C5G7S_SLR1e-3_4000_095_HBC_nc4096_nb512_nt[120, 120]_ns500000_nn[2, 64, 64, 64, 64, 64, 6]_Tanh',
            'Scaled_Mixed_FCN_2G_C5G7S_SLR1e-3_4000_095_HBC_nc5120_nb512_nt[120, 120]_ns500000_nn[2, 64, 64, 64, 64, 64, 6]_Tanh',
            'Scaled_Mixed_FCN_2G_C5G7S_SLR1e-3_4000_095_HBC_nc10240_nb512_nt[120, 120]_ns500000_nn[2, 64, 64, 64, 64, 64, 6]_Tanh',
            'EVO_Scaled_Mixed_FCN_2G_C5G7S_SLR1e-3_4000_095_HBC_nc4096_nb512_nt[120, 120]_ns500000_nn[2, 64, 64, 64, 64, 64, 6]_Tanh',
            'EVO_Scaled_Mixed_FCN_2G_C5G7S_SLR1e-3_4000_095_HBC_nc5120_nb512_nt[120, 120]_ns500000_nn[2, 64, 64, 64, 64, 64, 6]_Tanh',
            'EVO_Scaled_Mixed_FCN_2G_C5G7S_SLR1e-3_4000_095_HBC_nc10240_nb512_nt[120, 120]_ns500000_nn[2, 64, 64, 64, 64, 64, 6]_Tanh',
        
        ]


legend=[r'Fixed $N=4096$',r'Fixed $N=5120$',r'Fixed $N=10240$',r'EVO $N=4096$',r'EVO $N=5120$',r'EVO $N=10240$'] 




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
plt.ylim([1.e-3,5])
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' + type_loss +ext)
plt.show()


## residual 


# list_ResFlux = []
# list_ResKeff = []
# list_AccKeff = []

# for Folder in Folders:
#     df_ResFlux = pd.read_csv( results_dir+Folder+'/ResFlux.csv')
#     df_ResKeff = pd.read_csv( results_dir+Folder+'/ResKeff.csv')
#     df_AccKeff = pd.read_csv( results_dir+Folder+'/AccKeff.csv')

#     list_ResFlux.append(df_ResFlux)
#     list_ResKeff.append(df_ResKeff)
#     list_AccKeff.append(df_AccKeff)

# fig, ax = plt.subplots()
# for df in list_ResFlux:
#     df.plot(x='Iteration',y='Value',ax=ax,logy=True)
# plt.legend(legend,ncol=2)
# fig.tight_layout()
# plt.savefig(save_dir+ test_case + '_' +'ResFlux'+ext)
# plt.show()

# fig, ax = plt.subplots()
# for df in list_ResKeff:
#     df.plot(x='Iteration',y='Value',ax=ax,logy=True)
# plt.legend(legend,ncol=2)
# fig.tight_layout()
# plt.savefig(save_dir+ test_case + '_' +'ResKeff'+ext)
# plt.show()

# fig, ax = plt.subplots()
# for df in list_AccKeff:
#     df.plot(x='Iteration',y='Value',ax=ax,logy=True)
# plt.legend(legend,ncol=2)
# fig.tight_layout()
# plt.savefig(save_dir+ test_case + '_' +'AccKeff'+ext)
# plt.show()
    