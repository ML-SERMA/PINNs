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
parser.add_argument('-t','--test', type=str, help="name of test case",default='IAEA_5S')
parser.add_argument('-l','--loss', type=str, help="name of loss",default='Loss')
args = parser.parse_args()


test_case= args.test 
Error_test = True
type_loss = args.loss

save_dir = test_case + '/'
os.makedirs(save_dir, exist_ok=True)
print(f"==>> save_dir: {save_dir}")



Folders = [ 'Mixed_FCN_FA_IAEA_5S_SLR1e-3_2000_095_nc10240_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'Mixed_FCN_FA_IAEA_5S_SLR1e-3_2000_095_adapt_625_64_nc2048_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'Mixed_FCN_FA_IAEA_5S_SLR1e-3_2000_095_adapt_1250_128_nc2048_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
            'Mixed_FCN_FA_IAEA_5S_SLR1e-3_2000_095_adapt_2500_256_nc2048_nb512_nt[96, 86]_ns200000_nn[2, 64, 64, 64, 64, 64, 3]_Tanh',
    
        ]

list_data = []

for Folder in Folders:
    pathFile = results_dir+'paper/IAEA-5S/'+Folder+'/Loss.csv'
    if Error_test:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_Test','Error_p'])
    else:
        df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    list_data.append(df)



legend=['FS','AS_64','AS_128','AS_256']   
fig, ax = plt.subplots()
for df in list_data:
    df.plot(x='Iter',y=type_loss,ax=ax,logy=True)
plt.legend(legend)
plt.savefig(save_dir+ test_case + '_' + type_loss +'.png')
plt.show()