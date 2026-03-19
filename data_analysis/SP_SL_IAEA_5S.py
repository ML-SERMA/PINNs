import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd



project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


results_dir =  project_dir + '/results/'+'compare_IAEA_5S_scaling_loss/'
data_dir = project_dir+'/data/'

print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")


parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('-t','--test', type=str, help="name of test case",default='IAEA-5S_newGPU')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Error_p_global')
args = parser.parse_args()


test_case= args.test 
Error_phi = True
type_loss = args.loss

save_dir = 'SP_'+test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")





Folders = [ 'Mixed_FCN_IAEA_5S_SLR2e-4_2000_095_HBC_nc10240_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh_random_Unscaled',
            'Mixed_FCN_IAEA_5S_SLR2e-4_2000_095_HBC_nc10240_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh_random_Scaled',
            'Mixed_FCN_IAEA_5S_SLR2e-4_2000_095_HBC_nc10240_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Sin_Sobol_Unscaled',
            'Mixed_FCN_IAEA_5S_SLR2e-4_2000_095_HBC_nc10240_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Sin_Sobol_Scaled',
            ]

legend=['UL-Tanh-Random','SL-Tanh-Random','UL-Sin-Sobol','SL-Sin-Sobol'] 










list_data = []

for Folder in Folders:
    pathFile = results_dir+Folder+'/Loss.csv'
    if Error_phi:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE',
                                                                    'Error_phi_global','Error_phi_g0',
                                                                    'Error_p_global','Error_p_g0'])

        # df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE',
        #                                                             'Error_phi_global','Error_phi_g0', 'Error_phi_g1',
        #                                                             'Error_p_global','Error_p_g0', 'Error_p_g1'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    list_data.append(df)




ext = '.pdf'
fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True)
plt.legend(legend,ncol=2)
plt.xlabel('Iteration')
# plt.ylim([1.e-3,5])
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
#             r'Scaled Loss for $\phi$',r'Scaled Loss for $\phi^1$',r'Scaled Loss for $\phi^2$']
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






# ## residual 
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
    