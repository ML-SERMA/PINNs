import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import matplotlib.colors as colors
import pyvista as pv
import torch

class visualization:
    @staticmethod
    def viewGrid(mesh):
        if mesh.ndim == 2:
            xs = np.asarray(mesh.xpos[0])
            ys = np.asarray(mesh.xpos[1])
            plt.figure()
            ax = plt.gca()
            # grid "shades" (boxes)
            #w, h = xs[1] - xs[0], ys[1] - ys[0]
            w = mesh.pas[0]
            h = mesh.pas[1]
            for i, x in enumerate(xs[:-1]):
                for j, y in enumerate(ys[:-1]):
                    #if i % 2 == j % 2:  # racing flag style
                    ax.add_patch(Rectangle((x, y), w[i], h[j], fill=True, color='#008610', alpha=.2))
            # grid lines
            for x in xs:
                plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle='-')
            for y in ys:
                plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle='-')
            plt.show(block=False)
    @staticmethod
    def viewFluxOnCell(Flux,mesh):
        maxFlux = np.max(Flux) 
        minFlux = np.min(Flux) 
        plt.figure()
        ax = plt.gca()

        Z = np.reshape(Flux, (mesh.nmail[1], mesh.nmail[0] ))
        ax.pcolormesh(mesh.xpos[0], mesh.xpos[1], Z, shading='flat', vmin=minFlux, vmax=maxFlux,cmap='jet')
        # plt.show(block=False)
        plt.show()
        
        
    @staticmethod
    def viewCrossSection(X,S,pathFile=None):
        x=X[:,0:1]
        y=X[:,1:2]
        if S is None:
            fig, ax = plt.subplots()
            sc = ax.scatter(x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), alpha=0.5,cmap='rainbow')
            ax.grid(True)
            plt.colorbar(sc)
            if pathFile is not None:
                plt.savefig(pathFile)
        else:
            fig, ax = plt.subplots()
            sc = ax.scatter(x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), c=S.cpu().detach().numpy(), alpha=0.5,cmap='rainbow')
            ax.grid(True)
            plt.colorbar(sc)
            if pathFile is not None:
                plt.savefig(pathFile)
    
    @staticmethod
    def viewErrorAndSolutionScale(params_domain,n_test,error,u_pred,u_test,save_dir,name='error_test.pdf',vminE=None,vmaxE=None,vminF=None,vmaxF=None):
        x_bound_low = params_domain['xmin'][0]
        x_bound_up  = params_domain['xmax'][0]
        y_bound_low = params_domain['xmin'][1]
        y_bound_up  = params_domain['xmax'][1]
        num_test_x = n_test[0]
        num_test_y = n_test[1]
        normE = colors.LogNorm(vmin=max(vminE,1.e-9), vmax=vmaxE)
        # normF = colors.LogNorm(vmin=vminF, vmax=vmaxF)
        
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(19)

        # Plot the Error
        ax1 = plt.subplot(1, 3, 1)
        shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth',norm=normE, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw1)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        # plt.colorbar(shw1,ax=ax1)
        ax1.set_title("Absolute Error")


        # Plot the Predicted values for the NN
        ax2 = plt.subplot(1, 3, 2)
        shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw2)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        ax2.set_title("Predicted Solution")


        # Plot the exact solution
        ax3 = plt.subplot(1, 3, 3)
        shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw3)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$') 
        ax3.set_title("Reference Solution")
        plt.tight_layout()
        plt.savefig(save_dir+name)

        isSeparateFigure = True

        if isSeparateFigure:
            fig, ax = plt.subplots()
            shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth',norm=normE, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw1)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.tight_layout()
            # ax.set_title("Absolute Error")
            plt.savefig(save_dir+'AE_'+name)
            #=============================================================================================
            fig, ax = plt.subplots()
            shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw2)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.tight_layout() 
            # ax.set_title("Predicted Solution")
            plt.savefig(save_dir+'Predicted_'+name)

            fig, ax = plt.subplots()
            shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw3)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.tight_layout()
            # ax.set_title("Reference Solution")
            plt.savefig(save_dir+'Reference_'+name)
            
    @staticmethod
    def viewErrorAndSolution(params_domain,n_test,error,u_pred,u_test,save_dir,name='error_test.pdf'):
        x_bound_low = params_domain['xmin'][0]
        x_bound_up  = params_domain['xmax'][0]
        y_bound_low = params_domain['xmin'][1]
        y_bound_up  = params_domain['xmax'][1]
        num_test_x = n_test[0]
        num_test_y = n_test[1]
        
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(19)

        # Plot the Error
        ax1 = plt.subplot(1, 3, 1)
        shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw1)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        # plt.colorbar(shw1,ax=ax1)
        ax1.set_title("Absolute Error")


        # Plot the Predicted values for the NN
        ax2 = plt.subplot(1, 3, 2)
        shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw2)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        ax2.set_title("Predicted Solution")


        # Plot the exact solution
        ax3 = plt.subplot(1, 3, 3)
        shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw3)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        ax3.set_title("Reference Solution")
        plt.tight_layout()
        plt.savefig(save_dir+name)

        isSeparateFigure = True

        if isSeparateFigure:
            fig, ax = plt.subplots()
            shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw1)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.tight_layout()
            # ax.set_title("Absolute Error")
            plt.savefig(save_dir+'AE_'+name)
            #=============================================================================================
            fig, ax = plt.subplots()
            shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw2)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.tight_layout() 
            # ax.set_title("Predicted Solution")
            plt.savefig(save_dir+'Predicted_'+name)

            fig, ax = plt.subplots()
            shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw3)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.tight_layout()
            # ax.set_title("Reference Solution")
            plt.savefig(save_dir+'Reference_'+name)



    @staticmethod
    def show_loss(pathFile,save_dir,Error_phi=False,Error_p=False):
        if Error_phi and Error_p:
            df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi','Error_p'])
            fig, axes = plt.subplots(nrows=1, ncols=3)
            df.plot(x='Iter',y='Loss',ax=axes[0],loglog=True)
            df.plot(x='Iter',y='Loss_BC',ax=axes[1],loglog=True)
            df.plot(x='Iter',y='Loss_PDE',ax=axes[2],loglog=True)
            plt.savefig(save_dir+'loss_train.png')

            fig, axes = plt.subplots(nrows=1, ncols=2)
            df.plot(x='Iter',y='Error_phi',ax=axes[0],loglog=True)
            df.plot(x='Iter',y='Error_p',ax=axes[1],loglog=True)
            plt.savefig(save_dir+'loss_test.png')


        elif Error_phi and not Error_p:
            df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi'])
            fig, axes = plt.subplots(nrows=2, ncols=2)
            # Plot the Error
            df.plot(x='Iter',y='Loss',ax=axes[0,0],loglog=True)
            df.plot(x='Iter',y='Loss_BC',ax=axes[0,1],loglog=True)
            df.plot(x='Iter',y='Loss_PDE',ax=axes[1,0],loglog=True)
            df.plot(x='Iter',y='Error_phi',ax=axes[1,1],loglog=True)
            plt.savefig(save_dir+'loss_train_test.png')
        else:
            df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
            fig, axes = plt.subplots(nrows=1, ncols=3)
            fig.set_figheight(5)
            fig.set_figwidth(19)
            # Plot the Error
            df.plot(x='Iter',y='Loss',ax=axes[0],loglog=True)
            df.plot(x='Iter',y='Loss_BC',ax=axes[1],loglog=True)
            df.plot(x='Iter',y='Loss_PDE',ax=axes[2],loglog=True)
            plt.savefig(save_dir+'loss_train.png')
    @staticmethod
    def show_history_residuals(pathFile,save_dir,keff_ref_exist=False):

        if keff_ref_exist:
            df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','keff', 'resKeff', 'resFlux','innerSolver','resIS','accKeff'])
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
            axes = axes.flatten()
            df.plot(x='Iter', y='keff', ax=axes[0], loglog=True, title='keff')
            df.plot(x='Iter', y='accKeff', ax=axes[1], loglog=True, title='Accuracy keff')
            df.plot(x='Iter', y='resKeff', ax=axes[2], loglog=True, title='Residual keff')
            df.plot(x='Iter', y='resFlux', ax=axes[3], loglog=True, title='Residual Flux')
            df.plot(x='Iter', y='innerSolver', ax=axes[4], loglog=True, title='Inner Solver')
            df.plot(x='Iter', y='resIS', ax=axes[5], loglog=True, title='res IS')
            for ax in axes:
                ax.grid(True)
            fig.tight_layout()
            plt.savefig(save_dir + 'history_residuals_full.png')
            plt.close()
        else:
            df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','keff', 'resKeff', 'resFlux','innerSolver','resIS'])
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            axes = axes.flatten()
            df.plot(x='Iter', y='keff', ax=axes[0], loglog=True, title='keff')
            df.plot(x='Iter', y='resKeff', ax=axes[1], loglog=True, title='Residual keff')
            df.plot(x='Iter', y='resFlux', ax=axes[2], loglog=True, title='Residual Flux')
            df.plot(x='Iter', y='innerSolver', ax=axes[3], loglog=True, title='Inner Solver')
            for ax in axes:
                ax.grid(True)
            fig.tight_layout()
            plt.savefig(save_dir + 'history_residuals.png')
            plt.close()



    @staticmethod
    def viewSolution(params_domain,n_test,u,pathFile):
        ndim = len(params_domain['xmin'])
        if ndim ==2 :
            x_bound_low = params_domain['xmin'][0]
            x_bound_up  = params_domain['xmax'][0]
            y_bound_low = params_domain['xmin'][1]
            y_bound_up  = params_domain['xmax'][1]
            num_test_x = n_test[0]
            num_test_y = n_test[1]

            fig = plt.figure(frameon=False)
            shw1 = plt.imshow(u.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw1)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            plt.savefig(pathFile)
        if ndim == 3:
            x_bound_low = params_domain['xmin'][0]
            x_bound_up  = params_domain['xmax'][0]
            y_bound_low = params_domain['xmin'][1]
            y_bound_up  = params_domain['xmax'][1]
            z_bound_low = params_domain['xmin'][2]
            z_bound_up  = params_domain['xmax'][2]
            num_test_x = n_test[0]
            num_test_y = n_test[1]
            num_test_z = n_test[2]
            M = u.cpu().detach().numpy().reshape((num_test_x, num_test_y,num_test_z), order="F")
            index_z = n_test[2]//2
            M_slice = M[:,:,index_z]
            fig = plt.figure(frameon=False)
            shw1 = plt.imshow(M_slice.T, cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
            plt.colorbar(shw1)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            plt.savefig(pathFile)
        
    @staticmethod
    def viewEvolutionResidual(totalListResidual, paramFigure={'title':'','xlabel':'','ylabel':'','label':'','color':'blue'}):
        totalIter = [i for i in range(len(totalListResidual))]
        plt.figure()
        plt.plot(totalIter, totalListResidual, color=paramFigure['color'], linestyle='-', marker='s', linewidth=1, label=paramFigure['label'])

        plt.title(paramFigure['title'])
        plt.xlabel(paramFigure['xlabel'])
        plt.ylabel(paramFigure['ylabel'])
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        
        
    @staticmethod
    def viewErrorAndCurrents(params_domain,n_test,p_error,p_pred,p_test,save_dir,name='error_currents_test.png'):
        x_bound_low = params_domain['xmin'][0]
        x_bound_up  = params_domain['xmax'][0]
        y_bound_low = params_domain['xmin'][1]
        y_bound_up  = params_domain['xmax'][1]
        num_test_x = n_test[0]
        num_test_y = n_test[1]


        
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(20)

        # Plot the Error
        ax1 = plt.subplot(2, 3, 1)
        shw1 = plt.imshow(p_error[0].cpu().detach().numpy().reshape((num_test_y, num_test_x+1)), cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw1)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax1.set_title("Absolute Error")
        # Plot the Predicted values for the NN
        ax2 = plt.subplot(2, 3, 2)
        shw2 = plt.imshow(p_pred[0].cpu().detach().numpy().reshape((num_test_y, num_test_x+1)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw2)
        plt.xlabel('X')
        plt.ylabel('Y')    
        ax2.set_title("Predicted Solution")
        # Plot the exact solution
        ax3 = plt.subplot(2, 3, 3)
        shw3 = plt.imshow(p_test[0].cpu().detach().numpy().reshape((num_test_y, num_test_x+1)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw3)
        plt.xlabel('X')
        plt.ylabel('Y')  
        ax3.set_title("Reference Solution")
        #=================================================
        ax4 = plt.subplot(2, 3, 4)
        shw4 = plt.imshow(p_error[1].cpu().detach().numpy().reshape((num_test_y+1, num_test_x)), cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw4)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax4.set_title("Absolute Error")
        # Plot the Predicted values for the NN
        ax5 = plt.subplot(2, 3, 5)
        shw5 = plt.imshow(p_pred[1].cpu().detach().numpy().reshape((num_test_y+1, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw5)
        plt.xlabel('X')
        plt.ylabel('Y')    
        ax5.set_title("Predicted Solution")
        # Plot the exact solution
        
        ax6 = plt.subplot(2, 3, 6)
        shw6 = plt.imshow(p_test[1].cpu().detach().numpy().reshape((num_test_y+1, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
        plt.colorbar(shw6)
        plt.xlabel('X')
        plt.ylabel('Y')  
        ax6.set_title("Reference Solution")
        plt.savefig(save_dir+name)
        
    @staticmethod
    def export_flux_list(mesh, flux_list, filename="flux_data.vtr"):
        """
        Export multiple flux arrays on a structured Cartesian mesh.
        
        Args:
            mesh: Object with attribute `edges`, a list of 1D arrays for each dim (x_edges, y_edges, [z_edges]).
            flux_list: List of 2D or 3D flux arrays matching mesh cell count.
            filename: Output filename (should end with .vtr for 3D, .vts for 2D).
        """
        
        # Convert edges to NumPy arrays on CPU
        edges = []
        for e in mesh.edges:
            if torch.is_tensor(e):
                edges.append(e.detach().cpu().numpy())
            else:
                edges.append(np.asarray(e))

        ndims = len(edges)  # 2 or 3

        if ndims not in (2, 3):
            raise ValueError(f"Mesh edges dimension should be 2 or 3, got {ndims}")

        # Number of cells per dimension = len(edges[d]) - 1
        shape = tuple(len(e) - 1 for e in edges)  # e.g. (nx, ny, nz) or (nx, ny)

        # Create the PyVista structured grid (RectilinearGrid)
        if ndims == 2:
            # For 2D, add a singleton z dimension
            x_edges, y_edges = edges
            z_edges = np.array([0, 1])  # dummy edges in z to create 3D grid with thickness 1
            grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
        else:
            x_edges, y_edges, z_edges = edges
            grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)

        # Add each flux array as cell data
        for idx, flux in enumerate(flux_list, 1):
            # Convert flux to NumPy on CPU
            if torch.is_tensor(flux):
                flux_np = flux.detach().cpu().numpy()
            else:
                flux_np = np.asarray(flux)
                
            # Reshape flux to (nx, ny, [nz]) accordingly
            try:
                flux_reshaped = flux_np.reshape(shape, order='F')
            except Exception as e:
                raise ValueError(f"Flux index {idx} cannot be reshaped to mesh shape {shape}: {e}")

            # Flatten in Fortran order for VTK
            flux_flat = flux_reshaped.flatten(order='F')

            # Add to cell data with generic name
            name = f"flux_{idx}"
            grid.cell_data[name] = flux_flat

        # Save grid
        grid.save(filename)
        print(f"Saved {len(flux_list)} flux fields to '{filename}'")




