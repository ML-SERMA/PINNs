import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
project_dir  = os.path.dirname(os.path.dirname(__file__))
sys.path.append(str(project_dir))


from external_libraries.pyAMRNTE.Meshes.CartesianMesh import CartesianMesh
from external_libraries.pyAMRNTE.Finite_Element_Methods.Mixed_FEM import mixedFEM
from external_libraries.pyAMRNTE.Materials.geometry import geometry
from external_libraries.pyAMRNTE.Materials.crossSection import crossSection
from external_libraries.pyAMRNTE.Solvers.Mono_Solver import Mono_Solver
from external_libraries.pyAMRNTE.Tools.visualization import visualization
from external_libraries.pyAMRNTE.Tools.librarySave import saveResult
from external_libraries.pyAMRNTE.Tools.basic_utils import check_create_dir
from external_libraries.pyAMRNTE.Solvers.solution import solution

results_dir =  project_dir + '/results/'
data_dir = project_dir+'/data/'


print(f"==>> project_dir: {project_dir}")
print(f"==>> results_dir: {results_dir}")
print(f"==>> data_dir: {data_dir}")



params_mesh = {'ndim':2, 'xmin':[0, 0], 'xmax':[96, 86], 'nmail':[192, 172]}
params_fem  = {'ndim':2 , 'femOrder':[0, 0]}
params_solver = {'nIterPowerInverse':1000, 'epsilonKeff': 1.e-6, 'epsilonFlux':1.e-5}
# boundaryConditionType= [['ZERO_FLUX', 'ZERO_FLUX'],['ZERO_FLUX', 'ZERO_FLUX']]
boundaryConditionType= [['VACUUM', 'VACUUM'],['VACUUM', 'VACUUM']]
params_geometry = {'ndim':2, 'xmin':[0, 0], 'xmax':[1, 1], 'ngeometry':[3, 3]}
params_sigScat  = {'nameTestcase':'IAEA-5S'}

mesh = CartesianMesh(**params_mesh)
# mesh.plotMesh()
# mesh.viewGrid()
# print(mesh.xmid)

FEM = mixedFEM(**params_fem)
myGeometry = geometry(**params_geometry)

mySigScat = crossSection(mesh,myGeometry,nameTestCase=params_sigScat['nameTestcase'])
visualization.viewFluxOnCell(1/mySigScat.scatOdd,mesh)
visualization.viewFluxOnCell(mySigScat.scatEven,mesh)
visualization.viewFluxOnCell(mySigScat.scatFission,mesh)

nameFolder = 'IAEA-5S'
savedata_dir = check_create_dir(data_dir+nameFolder+'/')
isGenerateNewSolution = True
if isGenerateNewSolution:
    mySolution = Mono_Solver.generate_refSolution(mesh,FEM,myGeometry,boundaryConditionType,params_sigScat)
    mySolution.save_solution(n=192,folder_dir=savedata_dir)
visualization.viewFluxOnCell(mySolution.flux,mesh)
saveResult.saveFigures(savedata_dir,'Flux_192_172')