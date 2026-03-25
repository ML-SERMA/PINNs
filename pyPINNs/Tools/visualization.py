import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import matplotlib.colors as colors
# import pyvista as pv

import torch
import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import pyvista as pv

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
    def viewCrossSection(X, S=None, pathFile=None, title=None, cmap='rainbow', alpha=0.6, s=20, show=True):
        """
        Visualize cross-section (2D or 3D) or spatial point distribution.

        Parameters:
            X (Tensor): Coordinates [N, 2] or [N, 3] (x, y [, z]).
            S (Tensor or None): Values to color-code points. If None, no color.
            pathFile (str or None): If provided, saves figure to this path.
            title (str or None): Title of the plot.
            cmap (str): Colormap.
            alpha (float): Transparency of scatter points.
            s (int): Size of points.
            show (bool): Whether to show the plot (useful for batch saving).
        """
        X = X.detach().cpu()
        dim = X.shape[1]
        assert dim in (2, 3), f"Expected 2D or 3D coordinates, got {dim}D"

        if S is not None:
            S = S.detach().cpu()

        fig = plt.figure()
        
        if dim == 2:
            ax = fig.add_subplot(111)
            sc = ax.scatter(X[:, 0], X[:, 1], c=S if S is not None else 'gray',
                            cmap=cmap if S is not None else None,
                            alpha=alpha, s=s)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=S if S is not None else 'gray',
                            cmap=cmap if S is not None else None,
                            alpha=alpha, s=s)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        if S is not None:
            plt.colorbar(sc, ax=ax, label='Value')

        if title:
            ax.set_title(title)

        ax.grid(True)

        if pathFile:
            plt.savefig(pathFile, bbox_inches='tight')

        if show:
            plt.show()

        plt.close(fig)   
        
    # @staticmethod
    # def viewCrossSection(X,S,pathFile=None):
        
    #     x=X[:,0:1]
    #     y=X[:,1:2]
    #     if S is None:
    #         fig, ax = plt.subplots()
    #         sc = ax.scatter(x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), alpha=0.5,cmap='rainbow')
    #         ax.grid(True)
    #         plt.colorbar(sc)
    #         if pathFile is not None:
    #             plt.savefig(pathFile)
    #     else:
    #         fig, ax = plt.subplots()
    #         sc = ax.scatter(x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), c=S.cpu().detach().numpy(), alpha=0.5,cmap='rainbow')
    #         ax.grid(True)
    #         plt.colorbar(sc)
    #         if pathFile is not None:
    #             plt.savefig(pathFile)
    
    # @staticmethod
    # def viewErrorAndSolutionScale(params_domain,n_test,error,u_pred,u_test,save_dir,name='error_test.pdf',vminE=None,vmaxE=None,vminF=None,vmaxF=None):
    #     x_bound_low = params_domain['xmin'][0]
    #     x_bound_up  = params_domain['xmax'][0]
    #     y_bound_low = params_domain['xmin'][1]
    #     y_bound_up  = params_domain['xmax'][1]
    #     num_test_x = n_test[0]
    #     num_test_y = n_test[1]
    #     normE = colors.LogNorm(vmin=max(vminE,1.e-9), vmax=vmaxE)
    #     # normF = colors.LogNorm(vmin=vminF, vmax=vmaxF)
        
    #     fig = plt.figure()
    #     fig.set_figheight(5)
    #     fig.set_figwidth(19)

    #     # Plot the Error
    #     ax1 = plt.subplot(1, 3, 1)
    #     shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth',norm=normE, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #     plt.colorbar(shw1)
    #     plt.xlabel(r'$x_1$')
    #     plt.ylabel(r'$x_2$')
    #     # plt.colorbar(shw1,ax=ax1)
    #     ax1.set_title("Absolute Error")


    #     # Plot the Predicted values for the NN
    #     ax2 = plt.subplot(1, 3, 2)
    #     shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #     plt.colorbar(shw2)
    #     plt.xlabel(r'$x_1$')
    #     plt.ylabel(r'$x_2$')
    #     ax2.set_title("Predicted Solution")


    #     # Plot the exact solution
    #     ax3 = plt.subplot(1, 3, 3)
    #     shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #     plt.colorbar(shw3)
    #     plt.xlabel(r'$x_1$')
    #     plt.ylabel(r'$x_2$') 
    #     ax3.set_title("Reference Solution")
    #     plt.tight_layout()
    #     plt.savefig(save_dir+name)

    #     isSeparateFigure = True

    #     if isSeparateFigure:
    #         fig, ax = plt.subplots()
    #         shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth',norm=normE, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw1)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         plt.tight_layout()
    #         # ax.set_title("Absolute Error")
    #         plt.savefig(save_dir+'AE_'+name)
    #         #=============================================================================================
    #         fig, ax = plt.subplots()
    #         shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw2)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         plt.tight_layout() 
    #         # ax.set_title("Predicted Solution")
    #         plt.savefig(save_dir+'Predicted_'+name)

    #         fig, ax = plt.subplots()
    #         shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow',vmin=vminF,vmax=vmaxF, interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw3)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         plt.tight_layout()
    #         # ax.set_title("Reference Solution")
    #         plt.savefig(save_dir+'Reference_'+name)
            
    # @staticmethod
    # def viewErrorAndSolution(params_domain,n_test,error,u_pred,u_test,save_dir,name='error_test.pdf'):
    #     ndim = len(params_domain['xmin'])
    #     if ndim ==2 :
    #         x_bound_low = params_domain['xmin'][0]
    #         x_bound_up  = params_domain['xmax'][0]
    #         y_bound_low = params_domain['xmin'][1]
    #         y_bound_up  = params_domain['xmax'][1]
    #         num_test_x = n_test[0]
    #         num_test_y = n_test[1]
            
    #         fig = plt.figure()
    #         fig.set_figheight(5)
    #         fig.set_figwidth(19)

    #         # Plot the Error
    #         ax1 = plt.subplot(1, 3, 1)
    #         shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw1)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         # plt.colorbar(shw1,ax=ax1)
    #         ax1.set_title("Absolute Error")


    #         # Plot the Predicted values for the NN
    #         ax2 = plt.subplot(1, 3, 2)
    #         shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw2)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         ax2.set_title("Predicted Solution")


    #         # Plot the exact solution
    #         ax3 = plt.subplot(1, 3, 3)
    #         shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw3)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         ax3.set_title("Reference Solution")
    #         plt.tight_layout()
    #         plt.savefig(save_dir+name)

    #         isSeparateFigure = False
    #         if isSeparateFigure:
    #             fig, ax = plt.subplots()
    #             shw1 = plt.imshow(error.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #             plt.colorbar(shw1)
    #             plt.xlabel(r'$x_1$')
    #             plt.ylabel(r'$x_2$')
    #             plt.tight_layout()
    #             # ax.set_title("Absolute Error")
    #             plt.savefig(save_dir+'AE_'+name)
    #             #=============================================================================================
    #             fig, ax = plt.subplots()
    #             shw2 = plt.imshow(u_pred.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #             plt.colorbar(shw2)
    #             plt.xlabel(r'$x_1$')
    #             plt.ylabel(r'$x_2$')
    #             plt.tight_layout() 
    #             # ax.set_title("Predicted Solution")
    #             plt.savefig(save_dir+'Predicted_'+name)

    #             fig, ax = plt.subplots()
    #             shw3 = plt.imshow(u_test.cpu().detach().numpy().reshape((num_test_y, num_test_x)), cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #             plt.colorbar(shw3)
    #             plt.xlabel(r'$x_1$')
    #             plt.ylabel(r'$x_2$')
    #             plt.tight_layout()
    #             # ax.set_title("Reference Solution")
    #             plt.savefig(save_dir+'Reference_'+name)

    #     elif ndim ==3 :
    #         x_bound_low = params_domain['xmin'][0]
    #         x_bound_up  = params_domain['xmax'][0]
    #         y_bound_low = params_domain['xmin'][1]
    #         y_bound_up  = params_domain['xmax'][1]
    #         num_test_x = n_test[0]
    #         num_test_y = n_test[1]
    #         num_test_z = n_test[2]

    #         M_error = error.cpu().detach().numpy().reshape((num_test_x, num_test_y,num_test_z), order="F")
    #         M_pred = u_pred.cpu().detach().numpy().reshape((num_test_x, num_test_y,num_test_z), order="F")
    #         M_test = u_test.cpu().detach().numpy().reshape((num_test_x, num_test_y,num_test_z), order="F")

    #         index_z = n_test[2]//2
    #         M_error_slice = M_error[:,:,index_z]
    #         M_pred_slice  = M_pred[:,:,index_z]
    #         M_test_slice  = M_test[:,:,index_z]
            
    #         fig = plt.figure()
    #         fig.set_figheight(5)
    #         fig.set_figwidth(19)

    #         # Plot the Error
    #         ax1 = plt.subplot(1, 3, 1)
    #         shw1 = plt.imshow(M_error_slice.T, cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw1)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         # plt.colorbar(shw1,ax=ax1)
    #         ax1.set_title("Absolute Error")

    #         # Plot the Predicted values for the NN
    #         ax2 = plt.subplot(1, 3, 2)
    #         shw2 = plt.imshow(M_pred_slice.T, cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw2)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         ax2.set_title("Predicted Solution")

    #         # Plot the exact solution
    #         ax3 = plt.subplot(1, 3, 3)
    #         shw3 = plt.imshow(M_test_slice.T, cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=(x_bound_low, x_bound_up, y_bound_low, y_bound_up))
    #         plt.colorbar(shw3)
    #         plt.xlabel(r'$x_1$')
    #         plt.ylabel(r'$x_2$')
    #         ax3.set_title("Reference Solution")
    #         plt.tight_layout()
    #         plt.savefig(save_dir+name)
    #     else:
    #         print('Pass')

    # @staticmethod
    # def show_loss(pathFile,save_dir,Error_phi=False,Error_p=False):
    #     if Error_phi and Error_p:
    #         df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi','Error_p'])
    #         fig, axes = plt.subplots(nrows=1, ncols=3)
    #         df.plot(x='Iter',y='Loss',ax=axes[0],loglog=True)
    #         df.plot(x='Iter',y='Loss_BC',ax=axes[1],loglog=True)
    #         df.plot(x='Iter',y='Loss_PDE',ax=axes[2],loglog=True)
    #         plt.savefig(save_dir+'loss_train.png')

    #         fig, axes = plt.subplots(nrows=1, ncols=2)
    #         df.plot(x='Iter',y='Error_phi',ax=axes[0],loglog=True)
    #         df.plot(x='Iter',y='Error_p',ax=axes[1],loglog=True)
    #         plt.savefig(save_dir+'loss_test.png')


    #     elif Error_phi and not Error_p:
    #         df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE','Error_phi'])
    #         fig, axes = plt.subplots(nrows=2, ncols=2)
    #         # Plot the Error
    #         df.plot(x='Iter',y='Loss',ax=axes[0,0],loglog=True)
    #         df.plot(x='Iter',y='Loss_BC',ax=axes[0,1],loglog=True)
    #         df.plot(x='Iter',y='Loss_PDE',ax=axes[1,0],loglog=True)
    #         df.plot(x='Iter',y='Error_phi',ax=axes[1,1],loglog=True)
    #         plt.savefig(save_dir+'loss_train_test.png')
    #     else:
    #         df = pd.read_csv(pathFile, delimiter='\s+',skiprows=1,names=['Iter','Loss','Loss_BC','Loss_PDE'])
    #         fig, axes = plt.subplots(nrows=1, ncols=3)
    #         fig.set_figheight(5)
    #         fig.set_figwidth(19)
    #         # Plot the Error
    #         df.plot(x='Iter',y='Loss',ax=axes[0],loglog=True)
    #         df.plot(x='Iter',y='Loss_BC',ax=axes[1],loglog=True)
    #         df.plot(x='Iter',y='Loss_PDE',ax=axes[2],loglog=True)
    #         plt.savefig(save_dir+'loss_train.png')
    @staticmethod
    def show_loss(pathFile, save_dir, n_phi_groups=0, n_p_groups=0):
        """
        Show training losses and test errors.

        Args:
            pathFile: CSV log file
            save_dir: directory to save plots
            n_phi_groups: number of flux groups
            n_p_groups: number of currents groups
        """
        # --- Build column names ---
        header = ['Iter', 'Loss', 'Loss_BC', 'Loss_PDE']

        # Flux errors
        if n_phi_groups > 0:
            header.append('Error_phi_global')
            header.extend([f'Error_phi_g{g}' for g in range(n_phi_groups)])

        # Currents errors
        if n_p_groups > 0:
            header.append('Error_p_global')
            header.extend([f'Error_p_g{g}' for g in range(n_p_groups)])

        # --- Read CSV ---
        df = pd.read_csv(pathFile, delimiter='\s+', skiprows=1, names=header)

        # --- Plot training losses ---
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        df.plot(x='Iter', y='Loss', ax=axes[0], loglog=True)
        df.plot(x='Iter', y='Loss_BC', ax=axes[1], loglog=True)
        df.plot(x='Iter', y='Loss_PDE', ax=axes[2], loglog=True)
        plt.savefig(save_dir+'loss_train.png')

        # --- Plot flux errors if available ---
        if n_phi_groups > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            df.plot(x='Iter', y='Error_phi_global', ax=ax, loglog=True, label='phi_global')
            # Plot per-group flux errors
            for g in range(n_phi_groups):
                df.plot(x='Iter', y=f'Error_phi_g{g}', ax=ax, loglog=True, label=f'phi_g{g}')
            plt.legend()
            plt.savefig(save_dir+'error_phi.png')

        # --- Plot currents errors if available ---
        if n_p_groups > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            df.plot(x='Iter', y='Error_p_global', ax=ax, loglog=True, label='p_global')
            for g in range(n_p_groups):
                df.plot(x='Iter', y=f'Error_p_g{g}', ax=ax, loglog=True, label=f'p_g{g}')
            plt.legend()
            plt.savefig(save_dir+'error_p.png')

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
        
        
    # @staticmethod
    # def viewErrorAndCurrents(mesh,p_error,p_pred,p_test,save_dir,name='error_currents_test.png'):
    #     ndim = mesh.ndim
    #     if ndim ==2 :
            
    #         xmid, ymid = mesh.xmid
    #         xpos, ypos = mesh.xpos
    #         nx, ny = len(xmid), len(ymid)
    #         bounds = (mesh.xmin[0],mesh.xmax[0], mesh.xmin[1], mesh.xmax[1]) # on 2D
            
    #         sizes = [(ny, nx+1),  (ny+1, nx)]
    #         P_error = [p_error[id].cpu().detach().numpy().reshape(sizes[id], order="C") for id in range(ndim)]
    #         P_pred =  [p_pred[id].cpu().detach().numpy().reshape(sizes[id], order="C") for id in range(ndim)]
    #         P_test =  [p_test[id].cpu().detach().numpy().reshape(sizes[id], order="C") for id in range(ndim)]

    #         fig = plt.figure()
    #         fig.set_figheight(10)
    #         fig.set_figwidth(20)

    #         for id in range(ndim):
    #             # Plot the Error
    #             ax1 = plt.subplot(2, 3, 3*id +1)
    #             shw1 = plt.imshow(P_error[id], cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=bounds)
    #             plt.colorbar(shw1)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax1.set_title("Absolute Error")
    #             # Plot the Predicted values for the NN
    #             ax2 = plt.subplot(2, 3, 3*id+2)
    #             shw2 = plt.imshow(P_pred[id], cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=bounds)
    #             plt.colorbar(shw2)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')    
    #             ax2.set_title("Predicted Solution")
    #             # Plot the exact solution
    #             ax3 = plt.subplot(2, 3, 3*id+3)
    #             shw3 = plt.imshow(P_test[id], cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=bounds)
    #             plt.colorbar(shw3)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')  
    #             ax3.set_title("Reference Solution")
    #         plt.savefig(save_dir+name)

    #     elif ndim==3:
            
    #         xmid, ymid, zmid = mesh.xmid
    #         xpos, ypos, zpos = mesh.xpos
    #         nx, ny, nz = len(xmid), len(ymid), len(zmid)
    #         bounds = (mesh.xmin[0],mesh.xmax[0], mesh.xmin[1], mesh.xmax[1]) # on 2D
            
            
    #         ##=====================================
    #         sizes = [(nx + 1, ny, nz),  (nx, ny + 1, nz), (nx, ny, nz + 1)]
    #         P_error = [p_error[id].cpu().detach().numpy().reshape(sizes[id], order="F") for id in range(ndim)]
    #         P_pred =  [p_pred[id].cpu().detach().numpy().reshape(sizes[id], order="F") for id in range(ndim)]
    #         P_test =  [p_test[id].cpu().detach().numpy().reshape(sizes[id], order="F") for id in range(ndim)]
    #         index_z = nz//2

    #         fig = plt.figure()
    #         fig.set_figheight(10)
    #         fig.set_figwidth(20)
    #         for id in range(ndim):
    #             ax1 = plt.subplot(3, 3, 3*id+1)
    #             shw1 = plt.imshow(P_error[id][:,:,index_z].T, cmap='gist_earth', interpolation="none", aspect='auto', origin='lower', extent=bounds)
    #             plt.colorbar(shw1)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax1.set_title("Absolute Error")
    #             # Plot the Predicted values for the NN
    #             ax2 = plt.subplot(3, 3, 3*id+2)
    #             shw2 = plt.imshow(P_pred[id][:,:,index_z].T, cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=bounds)
    #             plt.colorbar(shw2)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')    
    #             ax2.set_title("Predicted Solution")
    #             # Plot the exact solution
    #             ax3 = plt.subplot(3, 3, 3*id+3)
    #             shw3 = plt.imshow(P_test[id][:,:,index_z].T, cmap='rainbow', interpolation="none", aspect='auto', origin='lower', extent=bounds)
    #             plt.colorbar(shw3)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')  
    #             ax3.set_title("Reference Solution")
    #         plt.savefig(save_dir+name)


    # @staticmethod
    # def viewFlux(mesh, phi_pred, save_dir, name='flux_pred.png', order='C'):
    #     """
    #     Visualizes the scalar flux (phi_pred) for both 2D and 3D cases.
        
    #     Parameters:
    #     -----------
    #     mesh : object
    #         Mesh object with attributes: ndim, xmid, xmin, xmax
    #     phi_pred : torch.Tensor
    #         Predicted flux values, shape (N, 1) where N = nx*ny (2D) or nx*ny*nz (3D)
    #     save_dir : str
    #         Directory to save the plot
    #     name : str
    #         File name to save the figure
    #     order : str
    #         'C' or 'F' order used when reshaping for visualization
    #     """
    #     ndim = mesh.ndim
    #     os.makedirs(save_dir, exist_ok=True)

    #     if ndim == 2:
    #         centers = [center.detach().cpu().numpy() for center in mesh.centers]    
    #         xmid, ymid = centers
    #         xmin = mesh.xmin.detach().cpu().numpy()
    #         xmax = mesh.xmax.detach().cpu().numpy()
    #         nx, ny = xmid.shape[0], ymid.shape[0]
    #         bounds = (xmin[0], xmax[0], xmin[1], xmax[1])
    #         phi = phi_pred.cpu().detach().numpy().reshape((nx, ny), order='C')
    #         plt.figure(figsize=(8, 6))
    #         im = plt.imshow(phi.T, cmap='rainbow', interpolation="none", origin='lower', aspect='auto', extent=bounds)
    #         plt.colorbar(im)
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.title('Predicted Scalar Flux')
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(save_dir, name))
    #         plt.close()

    #     elif ndim == 3:
    #         centers = [center.detach().cpu().numpy() for center in mesh.centers]
    #         xmid, ymid, zmid = centers
    #         xmin = mesh.xmin.detach().cpu().numpy()
    #         xmax = mesh.xmax.detach().cpu().numpy()
    #         nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]
    #         bounds = (xmin[0], xmax[0], xmin[1], xmax[1])


    #         phi = phi_pred.cpu().detach().numpy().reshape((nx, ny, nz), order=order)
    #         index_z = nz // 2
    #         phi_slice = phi[:, :, index_z].T  # Transpose to align with (X, Y) for imshow

    #         plt.figure(figsize=(8, 6))
    #         im = plt.imshow(phi_slice, cmap='rainbow', interpolation="none", origin='lower', aspect='auto', extent=bounds)
    #         plt.colorbar(im)
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.title(f'Predicted Scalar Flux (Z-slice {index_z})')
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(save_dir, name))
    #         plt.close()

    #     else:
    #         raise ValueError("Only 2D and 3D are supported.")

    # @staticmethod
    # def viewErrorAndFlux(mesh, flux_error, flux_pred, flux_test, save_dir, name='flux_test.png', order='C'):
    #     """
    #     Visualize error, predicted, and reference scalar flux for 2D and 3D meshes.

    #     Parameters:
    #     - mesh: object with attributes ndim, xmin, xmax, xmid
    #     - flux_error, flux_pred, flux_test: torch tensors of shape (N,) or (num_nodes,)
    #     - save_dir: directory to save the figure
    #     - name: filename for the figure
    #     - order: 'C' or 'F' for numpy reshape order
    #     """

    #     ndim = mesh.ndim

    #     if ndim == 2:
    #         centers = [center.detach().cpu().numpy() for center in mesh.centers]    
    #         xmid, ymid = centers
    #         xmin = mesh.xmin.detach().cpu().numpy()
    #         xmax = mesh.xmax.detach().cpu().numpy()
    #         nx, ny = xmid.shape[0], ymid.shape[0]
    #         bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

    #         # Reshape arrays (assume nodal data size = nx * ny)
    #         error_np = flux_error.cpu().detach().numpy().reshape((nx, ny), order=order)
    #         pred_np = flux_pred.cpu().detach().numpy().reshape((nx, ny), order=order)
    #         test_np = flux_test.cpu().detach().numpy().reshape((nx, ny), order=order)

    #         # Transpose if Fortran order
    #         # if order == 'F':
    #         error_np = error_np.T
    #         pred_np = pred_np.T
    #         test_np = test_np.T

    #         fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    #         im0 = axs[0].imshow(error_np, cmap='gist_earth', origin='lower', extent=bounds, aspect='auto')
    #         axs[0].set_title('Absolute Error')
    #         axs[0].set_xlabel('X')
    #         axs[0].set_ylabel('Y')
    #         plt.colorbar(im0, ax=axs[0])

    #         im1 = axs[1].imshow(pred_np, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
    #         axs[1].set_title('Predicted Flux')
    #         axs[1].set_xlabel('X')
    #         axs[1].set_ylabel('Y')
    #         plt.colorbar(im1, ax=axs[1])

    #         im2 = axs[2].imshow(test_np, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
    #         axs[2].set_title('Reference Flux')
    #         axs[2].set_xlabel('X')
    #         axs[2].set_ylabel('Y')
    #         plt.colorbar(im2, ax=axs[2])

    #         plt.tight_layout()
    #         plt.savefig(save_dir + name)
    #         plt.close()

    #     elif ndim == 3:
            
    #         centers = [center.detach().cpu().numpy() for center in mesh.centers]
    #         xmid, ymid, zmid = centers
    #         xmin = mesh.xmin.detach().cpu().numpy()
    #         xmax = mesh.xmax.detach().cpu().numpy()
    #         nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]
    #         bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

    #         # Reshape nodal data
    #         error_np = flux_error.cpu().detach().numpy().reshape((nx, ny, nz), order=order)
    #         pred_np = flux_pred.cpu().detach().numpy().reshape((nx, ny, nz), order=order)
    #         test_np = flux_test.cpu().detach().numpy().reshape((nx, ny, nz), order=order)

    #         index_z = nz // 2  # middle slice along z

    #         slice_error = error_np[:, :, index_z]
    #         slice_pred = pred_np[:, :, index_z]
    #         slice_test = test_np[:, :, index_z]

    #         # Transpose for Fortran order
    #         if order == 'F':
    #             slice_error = slice_error.T
    #             slice_pred = slice_pred.T
    #             slice_test = slice_test.T

    #         fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    #         im0 = axs[0].imshow(slice_error, cmap='gist_earth', origin='lower', extent=bounds, aspect='auto')
    #         axs[0].set_title(f'Absolute Error (slice z={index_z})')
    #         axs[0].set_xlabel('X')
    #         axs[0].set_ylabel('Y')
    #         plt.colorbar(im0, ax=axs[0])

    #         im1 = axs[1].imshow(slice_pred, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
    #         axs[1].set_title(f'Predicted Flux (slice z={index_z})')
    #         axs[1].set_xlabel('X')
    #         axs[1].set_ylabel('Y')
    #         plt.colorbar(im1, ax=axs[1])

    #         im2 = axs[2].imshow(slice_test, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
    #         axs[2].set_title(f'Reference Flux (slice z={index_z})')
    #         axs[2].set_xlabel('X')
    #         axs[2].set_ylabel('Y')
    #         plt.colorbar(im2, ax=axs[2])

    #         plt.tight_layout()
    #         plt.savefig(save_dir + name)
    #         plt.close()



    # @staticmethod
    # def viewErrorAndCurrents(mesh, p_error, p_pred, p_test, save_dir, name='error_currents_test.png', order='C'):
    #     """
    #     Visualize error, predicted, and test currents for 2D and 3D meshes.

    #     Parameters:
    #     - mesh: object with attributes ndim, xmin, xmax, centers...
    #     - p_error, p_pred, p_test: list of tensors (length ndim) for each component current
    #     - save_dir: directory to save the figure
    #     - name: filename for the figure
    #     - order: 'C' or 'F' for numpy reshape order
    #     """

    #     ndim = mesh.ndim
    #     if ndim == 2:
    #         centers = [center.detach().cpu().numpy() for center in mesh.centers]
                
    #         xmid, ymid = centers
    #         xmin = mesh.xmin.detach().cpu().numpy()
    #         xmax = mesh.xmax.detach().cpu().numpy()
    #         nx, ny = xmid.shape[0], ymid.shape[0]
    #         bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

    #         # Sizes for current components:
    #         # sizes = [(ny, nx + 1), (ny + 1, nx)]  # note: ny first because image vertical axis is Y
    #         sizes = [(nx + 1,ny), (nx,ny + 1)]
            
    #         P_error = [p_error[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
    #         P_pred = [p_pred[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
    #         P_test = [p_test[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]

    #         fig = plt.figure(figsize=(20, 10))

    #         for i in range(ndim):
    #             # transpose if order=='F', else no transpose
    #             # data_error = P_error[i].T if order == 'F' else P_error[i]
    #             # data_pred = P_pred[i].T if order == 'F' else P_pred[i]
    #             # data_test = P_test[i].T if order == 'F' else P_test[i]

    #             data_error = P_error[i].T 
    #             data_pred = P_pred[i].T 
    #             data_test = P_test[i].T 

    #             ax1 = plt.subplot(2, 3, 3 * i + 1)
    #             im1 = plt.imshow(data_error, cmap='gist_earth', interpolation="none", aspect='auto',
    #                             origin='lower', extent=bounds)
    #             plt.colorbar(im1)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax1.set_title("Absolute Error")

    #             ax2 = plt.subplot(2, 3, 3 * i + 2)
    #             im2 = plt.imshow(data_pred, cmap='rainbow', interpolation="none", aspect='auto',
    #                             origin='lower', extent=bounds)
    #             plt.colorbar(im2)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax2.set_title("Predicted Solution")

    #             ax3 = plt.subplot(2, 3, 3 * i + 3)
    #             im3 = plt.imshow(data_test, cmap='rainbow', interpolation="none", aspect='auto',
    #                             origin='lower', extent=bounds)
    #             plt.colorbar(im3)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax3.set_title("Reference Solution")

    #         plt.savefig(save_dir + name)
    #         plt.close()

    #     elif ndim == 3:
            
    #         centers = [center.detach().cpu().numpy() for center in mesh.centers]    
    #         xmid, ymid, zmid = centers
    #         xmin = mesh.xmin.detach().cpu().numpy()
    #         xmax = mesh.xmax.detach().cpu().numpy()
    #         nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]
    #         bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

    #         # Sizes for 3D current components
    #         sizes = [(nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1)]

    #         P_error = [p_error[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
    #         P_pred = [p_pred[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
    #         P_test = [p_test[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]

    #         index_z = nz // 2  # slice in the middle along z-axis

    #         fig = plt.figure(figsize=(20, 10))

    #         for i in range(ndim):
    #             # Slice at fixed z-index and transpose if needed
    #             slice_error = P_error[i][:, :, index_z]
    #             slice_pred = P_pred[i][:, :, index_z]
    #             slice_test = P_test[i][:, :, index_z]

    #             # Transpose for order='F', else no transpose
    #             if order == 'F':
    #                 slice_error = slice_error.T
    #                 slice_pred = slice_pred.T
    #                 slice_test = slice_test.T

    #             ax1 = plt.subplot(3, 3, 3 * i + 1)
    #             im1 = plt.imshow(slice_error, cmap='gist_earth', interpolation="none", aspect='auto',
    #                             origin='lower', extent=bounds)
    #             plt.colorbar(im1)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax1.set_title("Absolute Error")

    #             ax2 = plt.subplot(3, 3, 3 * i + 2)
    #             im2 = plt.imshow(slice_pred, cmap='rainbow', interpolation="none", aspect='auto',
    #                             origin='lower', extent=bounds)
    #             plt.colorbar(im2)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax2.set_title("Predicted Solution")

    #             ax3 = plt.subplot(3, 3, 3 * i + 3)
    #             im3 = plt.imshow(slice_test, cmap='rainbow', interpolation="none", aspect='auto',
    #                             origin='lower', extent=bounds)
    #             plt.colorbar(im3)
    #             plt.xlabel('X')
    #             plt.ylabel('Y')
    #             ax3.set_title("Reference Solution")

    #         plt.savefig(save_dir + name)
    #         plt.close()

    ####################################################################################################
    @staticmethod
    def viewFlux(mesh, phi_pred, save_dir, name='flux_pred.png', index='ij'):
        """
        Visualizes the scalar flux (phi_pred) for both 2D and 3D cases.

        Parameters:
        -----------
        mesh : object
            Mesh object with attributes: ndim, xmid, xmin, xmax
        phi_pred : torch.Tensor
            Predicted flux values, shape (N, 1) where N = nx*ny (2D) or nx*ny*nz (3D)
        save_dir : str
            Directory to save the plot
        name : str
            File name to save the figure
        index : str
            'ij' or 'xy', determines reshape and transpose for plotting
        """
    

        ndim = mesh.ndim
        os.makedirs(save_dir, exist_ok=True)

        if ndim == 2:
            centers = [c.detach().cpu().numpy() for c in mesh.centers]    
            xmid, ymid = centers
            nx, ny = xmid.shape[0], ymid.shape[0]
            xmin = mesh.xmin.detach().cpu().numpy()
            xmax = mesh.xmax.detach().cpu().numpy()
            bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

            # Determine reshape order
            if index == 'ij':
                phi = phi_pred.cpu().detach().numpy().reshape((nx, ny), order='F')
                phi_plot = phi.T  # transpose for imshow
            elif index == 'xy':
                phi = phi_pred.cpu().detach().numpy().reshape((ny, nx), order='C')
                phi_plot = phi  # no transpose needed for xy
            else:
                raise ValueError("index must be 'ij' or 'xy'")

            plt.figure(figsize=(8, 6))
            im = plt.imshow(phi_plot, cmap='rainbow', interpolation="none",
                            origin='lower', aspect='auto', extent=bounds)
            plt.colorbar(im)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Predicted Scalar Flux')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, name))
            plt.close()

        elif ndim == 3:
            centers = [c.detach().cpu().numpy() for c in mesh.centers]
            xmid, ymid, zmid = centers
            nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]
            xmin = mesh.xmin.detach().cpu().numpy()
            xmax = mesh.xmax.detach().cpu().numpy()
            bounds = (xmin[0], xmax[0], xmin[1], xmax[1])
            index_z = nz // 2

            if index == 'ij':
                phi = phi_pred.cpu().detach().numpy().reshape((nx, ny, nz), order='F')
                phi_slice = phi[:, :, index_z].T
            elif index == 'xy':
                phi = phi_pred.cpu().detach().numpy().reshape((ny, nx, nz), order='C')
                phi_slice = phi[:, :, index_z]  # no transpose needed
            else:
                raise ValueError("index must be 'ij' or 'xy'")

            plt.figure(figsize=(8, 6))
            im = plt.imshow(phi_slice, cmap='rainbow', interpolation="none",
                            origin='lower', aspect='auto', extent=bounds)
            plt.colorbar(im)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Predicted Scalar Flux (Z-slice {index_z})')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, name))
            plt.close()

        else:
            raise ValueError("Only 2D and 3D are supported.")

    @staticmethod
    def viewErrorAndFlux(mesh, flux_error, flux_pred, flux_test, save_dir, name='flux_test.png', index='ij'):
        """
        Visualize error, predicted, and reference scalar flux for 2D and 3D meshes.

        Parameters:
        -----------
        mesh : object
            Mesh object with attributes ndim, xmin, xmax, xmid
        flux_error, flux_pred, flux_test : torch.Tensor
            Tensors of shape (N,) where N = nx*ny (2D) or nx*ny*nz (3D)
        save_dir : str
            Directory to save the figure
        name : str
            Filename for the figure
        index : str
            'ij' or 'xy' for meshgrid convention
        """
       
        ndim = mesh.ndim
        os.makedirs(save_dir, exist_ok=True)

        if ndim == 2:
            centers = [c.detach().cpu().numpy() for c in mesh.centers]
            xmid, ymid = centers
            nx, ny = xmid.shape[0], ymid.shape[0]
            xmin = mesh.xmin.detach().cpu().numpy()
            xmax = mesh.xmax.detach().cpu().numpy()
            bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

            # Determine reshape based on meshgrid indexing
            if index == 'ij':
                shape = (nx, ny)
                transpose_plot = True
                order = 'F'
            elif index == 'xy':
                shape = (ny, nx)
                transpose_plot = False
                order = 'C'
            else:
                raise ValueError("index must be 'ij' or 'xy'")
            
            error_np = flux_error.cpu().detach().numpy().reshape(shape, order=order)
            pred_np = flux_pred.cpu().detach().numpy().reshape(shape, order=order)
            test_np = flux_test.cpu().detach().numpy().reshape(shape, order=order)
            if transpose_plot:
                error_np = error_np.T
                pred_np = pred_np.T
                test_np = test_np.T


            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axs[0].imshow(error_np, cmap='gist_earth', origin='lower', extent=bounds, aspect='auto')
            axs[0].set_title('Absolute Error')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            plt.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(pred_np, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
            axs[1].set_title('Predicted Flux')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            plt.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(test_np, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
            axs[2].set_title('Reference Flux')
            axs[2].set_xlabel('X')
            axs[2].set_ylabel('Y')
            plt.colorbar(im2, ax=axs[2])

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, name))
            plt.close()

        elif ndim == 3:
            centers = [c.detach().cpu().numpy() for c in mesh.centers]
            xmid, ymid, zmid = centers
            nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]
            xmin = mesh.xmin.detach().cpu().numpy()
            xmax = mesh.xmax.detach().cpu().numpy()
            bounds = (xmin[0], xmax[0], xmin[1], xmax[1])
            index_z = nz // 2

            # Determine reshape based on meshgrid indexing
            if index == 'ij':
                shape = (nx, ny, nz)
                transpose_plot = True
                order = 'F'    
            elif index == 'xy':
                shape = (ny, nx, nz)
                transpose_plot = False
                order = 'C'
            else:
                raise ValueError("index must be 'ij' or 'xy'")
            
            error_np = flux_error.cpu().detach().numpy().reshape(shape, order=order)
            pred_np = flux_pred.cpu().detach().numpy().reshape(shape, order=order)
            test_np = flux_test.cpu().detach().numpy().reshape(shape, order=order)
            # Take middle slice along z
            slice_error = error_np[:, :, index_z]
            slice_pred = pred_np[:, :, index_z]
            slice_test = test_np[:, :, index_z]
            if transpose_plot:
                slice_error = slice_error.T
                slice_pred = slice_pred.T
                slice_test = slice_test.T

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axs[0].imshow(slice_error, cmap='gist_earth', origin='lower', extent=bounds, aspect='auto')
            axs[0].set_title(f'Absolute Error (slice id_z={index_z})')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            plt.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(slice_pred, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
            axs[1].set_title(f'Predicted Flux (slice id_z={index_z})')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            plt.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(slice_test, cmap='rainbow', origin='lower', extent=bounds, aspect='auto')
            axs[2].set_title(f'Reference Flux (slice id_z={index_z})')
            axs[2].set_xlabel('X')
            axs[2].set_ylabel('Y')
            plt.colorbar(im2, ax=axs[2])

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, name))
            plt.close()

        else:
            raise ValueError("Only 2D and 3D are supported.")



    @staticmethod
    def viewErrorAndCurrents(mesh, p_error, p_pred, p_test, save_dir, name='error_currents_test.png', index='ij'):
        """
        Visualize error, predicted, and reference currents for 2D and 3D meshes,
        supporting both 'ij' and 'xy' meshgrid conventions.

        Parameters:
        -----------
        mesh : object
            Mesh object with attributes ndim, xmin, xmax, centers
        p_error, p_pred, p_test : list of torch.Tensors
            Each list has length = ndim (components of current), each tensor shape = num_nodes or nodes per component
        save_dir : str
            Directory to save the figure
        name : str
            Filename for the figure
        index : str
            'ij' or 'xy' for meshgrid convention
        """
    
        ndim = mesh.ndim
        os.makedirs(save_dir, exist_ok=True)

        if ndim == 2:
            centers = [c.detach().cpu().numpy() for c in mesh.centers]
            xmid, ymid = centers
            nx, ny = xmid.shape[0], ymid.shape[0]
            xmin = mesh.xmin.detach().cpu().numpy()
            xmax = mesh.xmax.detach().cpu().numpy()
            bounds = (xmin[0], xmax[0], xmin[1], xmax[1])

            # Determine sizes and transpose based on meshgrid indexing
            if index == 'ij':
                sizes = [(nx + 1, ny), (nx, ny + 1)]
                transpose_plot = True
                order = 'F'

            elif index == 'xy':
                sizes = [(ny, nx + 1), (ny + 1, nx)]
                transpose_plot = False
                order = 'C'
            else:
                raise ValueError("index must be 'ij' or 'xy'")

             # Reshape each component
            P_error = [p_error[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
            P_pred = [p_pred[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
            P_test = [p_test[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
            fig = plt.figure(figsize=(20, 10))
            for i in range(ndim):
                data_error = P_error[i].T if transpose_plot else P_error[i]
                data_pred = P_pred[i].T if transpose_plot else P_pred[i]
                data_test = P_test[i].T if transpose_plot else P_test[i]

                ax1 = plt.subplot(2, 3, 3 * i + 1)
                im1 = plt.imshow(data_error, cmap='gist_earth', interpolation="none",
                                aspect='auto', origin='lower', extent=bounds)
                plt.colorbar(im1)
                ax1.set_title("Absolute Error")
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')

                ax2 = plt.subplot(2, 3, 3 * i + 2)
                im2 = plt.imshow(data_pred, cmap='rainbow', interpolation="none",
                                aspect='auto', origin='lower', extent=bounds)
                plt.colorbar(im2)
                ax2.set_title("Predicted Solution")
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')

                ax3 = plt.subplot(2, 3, 3 * i + 3)
                im3 = plt.imshow(data_test, cmap='rainbow', interpolation="none",
                                aspect='auto', origin='lower', extent=bounds)
                plt.colorbar(im3)
                ax3.set_title("Reference Solution")
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, name))
            plt.close()

        elif ndim == 3:
            centers = [c.detach().cpu().numpy() for c in mesh.centers]
            xmid, ymid, zmid = centers
            nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]
            xmin = mesh.xmin.detach().cpu().numpy()
            xmax = mesh.xmax.detach().cpu().numpy()
            bounds = (xmin[0], xmax[0], xmin[1], xmax[1])
            index_z = nz // 2

            # Determine sizes based on meshgrid indexing
            if index == 'ij':
                sizes = [(nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1)]
                transpose_plot = True
                order = 'F'
            elif index == 'xy':
                sizes = [(ny, nx + 1, nz), (ny + 1, nx, nz), (ny, nx, nz + 1)]
                transpose_plot = False
                order = 'C'
            else:
                raise ValueError("index must be 'ij' or 'xy'")

            P_error = [p_error[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
            P_pred = [p_pred[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]
            P_test = [p_test[i].cpu().detach().numpy().reshape(sizes[i], order=order) for i in range(ndim)]

            fig = plt.figure(figsize=(20, 10))
            for i in range(ndim):
                # Take middle slice along z
                slice_error = P_error[i][:, :, index_z]
                slice_pred = P_pred[i][:, :, index_z]
                slice_test = P_test[i][:, :, index_z]

                if transpose_plot:
                    slice_error = slice_error.T
                    slice_pred = slice_pred.T
                    slice_test = slice_test.T

                ax1 = plt.subplot(3, 3, 3 * i + 1)
                im1 = plt.imshow(slice_error, cmap='gist_earth', interpolation="none",
                                aspect='auto', origin='lower', extent=bounds)
                plt.colorbar(im1)
                ax1.set_title("Absolute Error")
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')

                ax2 = plt.subplot(3, 3, 3 * i + 2)
                im2 = plt.imshow(slice_pred, cmap='rainbow', interpolation="none",
                                aspect='auto', origin='lower', extent=bounds)
                plt.colorbar(im2)
                ax2.set_title("Predicted Solution")
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')

                ax3 = plt.subplot(3, 3, 3 * i + 3)
                im3 = plt.imshow(slice_test, cmap='rainbow', interpolation="none",
                                aspect='auto', origin='lower', extent=bounds)
                plt.colorbar(im3)
                ax3.set_title("Reference Solution")
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, name))
            plt.close()

        else:
            raise ValueError("Only 2D and 3D are supported.")



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
        
        
    def viewFlux3D_volume(mesh, phi_pred, order='F', mode='volume', iso_value=None, show=True, save_path=None):
        """
        Visualize 3D scalar flux using PyVista.

        Parameters:
        -----------
        mesh : object
            Mesh object with attributes: xmid = [x, y, z], ndim, xmin, xmax
        phi_pred : torch.Tensor
            Flux tensor of shape (N, 1)
        order : str
            Reshape order: 'F' or 'C'
        mode : str
            'volume' for volume rendering or 'iso' for isosurface
        iso_value : float or None
            Required for 'iso' mode to define isosurface value
        show : bool
            If True, shows the plot interactively
        save_path : str or None
            If provided, saves the plot as HTML (WebGL viewer)
        """
        assert mesh.ndim == 3, "Only 3D visualization is supported."
        assert mode in ['volume', 'iso'], "mode must be 'volume' or 'iso'."

        # Mesh grid
        centers = [center.detach().cpu().numpy() for center in mesh.centers]    
        xmid, ymid, zmid = centers
        xmin = mesh.xmin.detach().cpu().numpy()
        xmax = mesh.xmax.detach().cpu().numpy()
        nx, ny, nz = xmid.shape[0], ymid.shape[0], zmid.shape[0]

        # Reshape phi
        phi_np = phi_pred.cpu().detach().numpy().reshape((nx, ny, nz), order=order)

        # Create a 3D grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(phi_np.shape) + 1  # one more than cells
        grid.origin = (xmin[0], xmin[1],xmin[2])  # bottom-left-front
        grid.spacing = (xmid[1] - xmid[0], ymid[1] - ymid[0], zmid[1] - zmid[0])     # dx, dy, dz

        # Attach the scalar field
        grid.cell_data["flux"] = phi_np.flatten(order=order)

        # Initialize plotter
        plotter = pv.Plotter()
        if mode == 'volume':
            plotter.add_volume(grid, scalars="flux", opacity="sigmoid", cmap="viridis")
        elif mode == 'iso':
            assert iso_value is not None, "iso_value must be specified for isosurface mode."
            contour = grid.contour([iso_value])
            plotter.add_mesh(contour, color='orange', opacity=0.75)

        plotter.add_axes()
        plotter.show_grid()
        if save_path:
            plotter.export_html(save_path)
        if show:
            plotter.show()

   


 
    


    def plot_flux_slices(mesh, flux_list, filename="flux.png", log_scale=False):
        """
        Visualize orthogonal slices using PyVista and save as PNG.

        Args:
            mesh: object with mesh.edges
            flux_list: list of flux arrays
            filename: output PNG file
            log_scale: apply log10 scaling
        """

        # --- Convert edges ---
        edges = []
        for e in mesh.edges:
            if torch.is_tensor(e):
                edges.append(e.detach().cpu().numpy())
            else:
                edges.append(np.asarray(e))

        ndims = len(edges)

        if ndims == 2:
            x_edges, y_edges = edges
            z_edges = np.array([0, 1])
        elif ndims == 3:
            x_edges, y_edges, z_edges = edges
        else:
            raise ValueError("Only 2D or 3D supported")

        # --- Create grid ---
        grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
        shape = tuple(len(e) - 1 for e in edges)

        # --- Add flux fields ---
        for idx, flux in enumerate(flux_list, 1):

            if torch.is_tensor(flux):
                flux_np = flux.detach().cpu().numpy()
            else:
                flux_np = np.asarray(flux)

            phi = flux_np.reshape(shape, order='F')

            if log_scale:
                phi = np.log10(phi + 1e-10)

            grid.cell_data[f"flux_{idx}"] = phi.flatten(order='F')

        
        pv.start_xvfb()
        plotter = pv.Plotter(off_screen=True)

        field = "flux_1"

        # Midpoints
        xmid = 0.5 * (x_edges[0] + x_edges[-1])
        ymid = 0.5 * (y_edges[0] + y_edges[-1])
        zmid = 0.5 * (z_edges[0] + z_edges[-1])

        # Orthogonal slices
        slices = grid.slice_orthogonal(x=xmid, y=ymid, z=zmid)

        plotter.add_mesh(
            slices,
            scalars=field,
            cmap="jet",
            show_scalar_bar=True,
        )

        plotter.add_mesh(grid.outline(), color="black")

        plotter.view_isometric()

        # --- Save PNG ---
        plotter.screenshot(filename)
        plotter.close()

        print(f"Saved figure: {filename}")



    # def plot_flux_isosurfaces(mesh, flux_list, filename="flux_iso.png", log_scale=False):
    #     """
    #     Visualize flux using isosurfaces (3D) and save as PNG.

    #     Args:
    #         mesh: object with mesh.edges
    #         flux_list: list of flux arrays
    #         filename: output PNG
    #         log_scale: apply log10 scaling
    #     """

    #     # --- Offscreen (cluster safe) ---
    #     os.environ["PYVISTA_OFF_SCREEN"] = "true"

    #     # --- Convert edges ---
    #     edges = []
    #     for e in mesh.edges:
    #         if torch.is_tensor(e):
    #             edges.append(e.detach().cpu().numpy())
    #         else:
    #             edges.append(np.asarray(e))

    #     ndims = len(edges)

    #     if ndims != 3:
    #         raise ValueError("Isosurfaces require 3D mesh")

    #     x_edges, y_edges, z_edges = edges

    #     # --- Create grid ---
    #     grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    #     shape = tuple(len(e) - 1 for e in edges)

    #     # --- Use first flux 
    #     flux = flux_list[1]

    #     if torch.is_tensor(flux):
    #         flux_np = flux.detach().cpu().numpy()
    #     else:
    #         flux_np = np.asarray(flux)

    #     phi = flux_np.reshape(shape, order='F')

    #     # --- Log scale 
    #     if log_scale:
    #         phi = np.log10(phi + 1e-10)
    #         scalar_name = "flux_log"
    #     else:
    #         scalar_name = "flux"

    #     grid.cell_data[scalar_name] = phi.flatten(order='F')

    #     # --- Create plotter ---
    #     pv.start_xvfb()
    #     plotter = pv.Plotter(off_screen=True)

    #     # --- Choose isovalues ---
    #     if log_scale:
    #         iso_values = np.linspace(phi.max() - 6, phi.max(), 5)
    #     else:
    #         iso_values = np.linspace(phi.min(), phi.max(), 5)
    #     grid = grid.cell_data_to_point_data()
    #     # --- Compute isosurfaces ---
    #     contours = grid.contour(
    #         isosurfaces=iso_values,
    #         scalars=scalar_name
    #     )

    #     # --- Add to plot ---
    #     plotter.add_mesh(
    #         contours,
    #         cmap="viridis",
    #         opacity=0.6,
    #         show_scalar_bar=True
    #     )

    #     # Domain outline
    #     plotter.add_mesh(grid.outline(), color="black")

    #     plotter.view_isometric()

    #     # --- Save ---
    #     plotter.screenshot(filename)
    #     plotter.close()

    #     print(f"Saved isosurface figure: {filename}")


   

    

  

    