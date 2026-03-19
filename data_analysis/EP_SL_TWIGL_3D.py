import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/'+'compare_TWIGL_3D_scaling_loss/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='TWIGL_3D')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_p_global')
args = parser.parse_args()


test_case= args.test 
Error_phi = True
type_loss = args.loss

save_dir = 'EP_'+test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")





Folders = [
        'EP_Mixed_FCN_2G_TWIGL_3D_SLR2e-4_09_Nit2000_HBC_nc204800_nb512_nt[50, 50, 50]_ns8000000_nn[3, 64, 64, 64, 64, 64, 8]_Tanh_random_Unscaled',
        'EP_Mixed_FCN_2G_TWIGL_3D_SLR2e-4_09_Nit2000_HBC_nc204800_nb512_nt[50, 50, 50]_ns8000000_nn[3, 64, 64, 64, 64, 64, 8]_Tanh_random_Scaled',
        'EP_Mixed_FCN_2G_TWIGL_3D_SLR2e-4_09_Nit2000_HBC_nc204800_nb512_nt[50, 50, 50]_ns8000000_nn[3, 64, 64, 64, 64, 64, 8]_Sin_Sobol_Unscaled',
        'EP_Mixed_FCN_2G_TWIGL_3D_SLR2e-4_09_Nit2000_HBC_nc204800_nb512_nt[50, 50, 50]_ns8000000_nn[3, 64, 64, 64, 64, 64, 8]_Sin_Sobol_Scaled',    
        ]

legend=[ r'UL-Tanh-Random',r'SL-Tanh-Random',
        r'UL-Sin-Sobol',r'SL-Sin-Sobol',
        ] 

# legend=[ r'UL-20480',r'SL-20480',
#         r'UL-40960',r'SL-40960',
#         ] 


list_data = []

for Folder in Folders:
    pathFile = results_dir+Folder+'/Loss.csv'
    if Error_phi:
        # df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE',
        #                                                             'Error_phi_global','Error_phi_g0',
        #                                                             'Error_p_global','Error_p_g0'])

        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE',
                                                                    'Error_phi_global','Error_phi_g0', 'Error_phi_g1',
                                                                    'Error_p_global','Error_p_g0', 'Error_p_g1'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    list_data.append(df)




ext = '.pdf'
fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True,)
plt.legend(legend,ncol=2)
plt.xlabel('Iteration')
plt.ylim([5.e-3,1.0])
# plt.ylim([3.e-3,0.3])
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' + type_loss +ext)
plt.show()


# ext = '.pdf'
# fig, ax = plt.subplots()
# for df in list_data:
#     df.plot(x='Iter',y='Error_phi_global',ax=ax,logy=True)
#     df.plot(x='Iter',y='Error_phi_g0',ax=ax,logy=True)
#     df.plot(x='Iter',y='Error_phi_g1',ax=ax,logy=True)
# legend = [r'Unscaled Loss for $\phi$',r'Unscaled Loss for $\phi^1$',r'Unscaled Loss for $\phi^2$',
#         r'Scaled Loss for $\phi$',r'Scaled Loss for $\phi^1$',r'Scaled Loss for $\phi^2$']
# plt.legend(legend,ncol=2)
# plt.xlabel('Iteration')
# # plt.ylim([1.e-3,5])
# fig.tight_layout()
# plt.savefig(save_dir+ test_case + '_' + 'Error_phi' +ext)
# plt.show()


# ext = '.pdf'
# fig, ax = plt.subplots()
# for df in list_data:
#     df.plot(x='Iter',y='Error_p_global',ax=ax,logy=True)
#     df.plot(x='Iter',y='Error_p_g0',ax=ax,logy=True)
#     df.plot(x='Iter',y='Error_p_g1',ax=ax,logy=True)
# legend = [r'Unscaled Loss for $\phi$',r'Unscaled Loss for $\phi^1$',r'Unscaled Loss for $\phi^2$',
#             r'Scaled Loss for $\phi$',r'Scaled Loss for $\phi^1$',r'Scaled Loss for $\phi^2$']
# plt.legend(legend,ncol=2)
# plt.xlabel('Iteration')
# # plt.ylim([1.e-3,5])
# fig.tight_layout()
# plt.savefig(save_dir+ test_case + '_' + 'Error_p' +ext)
# plt.show()




list_df = []
for Folder in Folders:
    pathFile = results_dir+Folder+'/train.csv'
    df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','keff', 
                                                                'resKeff', 'resFlux','innerSolver','resIS','accKeff'])

    list_df.append(df)

ext = '.pdf'
type_res = 'accKeff'
fig, ax = plt.subplots()
for df in list_df:
    df.plot(x='Iter',y=type_res,ax=ax,logy=True)
plt.legend(legend,ncol=2)
plt.xlabel('Iteration')
# plt.ylim([1.e-4,5])
fig.tight_layout()
plt.savefig(save_dir+ test_case + '_' + type_res +ext)
plt.show()













