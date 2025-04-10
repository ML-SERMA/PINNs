import torch
import time
from tqdm import tqdm
import os
from ..Tools.visualization import visualization

def compute_l1_loss(w):
    return torch.abs(w).sum()

def compute_l2_loss(w):
    return torch.square(w).sum()

def train(pde_model,data,n_step=100,verbose=0, log_every=100, LossFile='Loss.dat',save_dir = '',**kwargs):

    train_on_batch = kwargs.get('train_on_batch',False)
    learning_rate_annealing = kwargs.get('learning_rate_annealing',False)
    weight_ridge =  kwargs.get('weight_ridge',0)
    weight_lasso =  kwargs.get('weight_lasso',0)
    weight_Sobolev = kwargs.get('weight_Sobolve',0)
    weight_regularization = kwargs.get('weight_regularization',0)
    weight_interface = kwargs.get('weight_interface',0)
    resampling_every = kwargs.get('resampling_every',0)
    normalization = kwargs.get('normalization',False)
    ngroup        = kwargs.get('ngroup',None)
    mode = kwargs.get('mode','w+')


    isLossData = kwargs.get('isLossData',False)
    if isLossData:
        print('Generate exp points !')
        data.generate_exp_points()


    # torch.autograd.set_detect_anomaly(True)
    tin=time.time()
    if verbose ==3 :
        for key, value in kwargs.items():
            print("%s == %s" % (key, value))
        print(f'n_step = {n_step} , log = {log_every}, lossFile = {LossFile}, savedir = {save_dir}')

    # LR annealing
    pde_model.beta = 0.8 
    pde_model.adaptive_constant = 1.0
    

    def closure():
        pde_model.optimizer.zero_grad()
        loss_bc  = pde_model.loss_BC(X_bc)
        loss_f = pde_model.loss_PDE(X_train)
        loss = pde_model.adaptive_constant*(loss_bc) + loss_f

        #============== Loss data =========================
        if isLossData:
            loss_data = pde_model.loss_EXP(data.X_exp,data.phi_exp)
            loss = loss + loss_data
        
        # Regularization
        if weight_lasso or weight_ridge:
            parameters = []
            for parameter in pde_model.model.parameters():
                parameters.append(parameter.view(-1))
        if weight_lasso:
            l1 = weight_lasso * compute_l1_loss(torch.cat(parameters))
            loss += l1
        if weight_ridge:
            l2 = weight_ridge * compute_l2_loss(torch.cat(parameters))
            loss += l2
        if weight_Sobolev:
            lS = weight_Sobolev * pde_model.Sobolev_norm(X_train)
            loss +=lS
        if weight_regularization:
            lReg = weight_regularization* pde_model.mass_regularization(X_train)
            loss = loss + lReg
        if weight_interface:
            loss_jump = weight_interface*pde_model.interface_jump(X_interface)
            # print(f"==>> loss_jump: {loss_jump}")
            loss = loss + loss_jump

        
        loss.backward(retain_graph=True)

        ###############################################################################################
        ### GRADIENT CLIPPING
        # torch.nn.utils.clip_grad_norm_(pde_model.model.parameters(), 10.) # improves training stability
        ###############################################################################################

        
        
        '# Evaluation of adaptive_constant'
        if learning_rate_annealing and pde_model.iter % 10 == 0 and pde_model.iter > 0:
            with torch.no_grad():   # To avoid OOM
                grad_f_list  = (torch.autograd.grad(loss_f , pde_model.model.linears[0].weight, retain_graph=True, create_graph=True)[0]).flatten()
                grad_bc_list = (torch.autograd.grad(loss_bc, pde_model.model.linears[0].weight, retain_graph=True, create_graph=True)[0]).flatten()

                for i in range(len(pde_model.model.layers)-2):
                    grad_f_list  = torch.cat((grad_f_list , (torch.autograd.grad(loss_f , pde_model.model.linears[i+1].weight, retain_graph=True, create_graph=True)[0]).flatten() ))
                    grad_bc_list = torch.cat((grad_bc_list, (torch.autograd.grad(loss_bc, pde_model.model.linears[i+1].weight, retain_graph=True, create_graph=True)[0]).flatten() ))
                    
                max_grad_res           = torch.max(torch.abs(grad_f_list))
                mean_grad_bcs          = torch.mean(torch.abs(grad_bc_list)) 
                # mean_grad_res          = torch.mean(torch.abs(grad_f_list))

                adaptive_constant_tmp  = max_grad_res / mean_grad_bcs
                pde_model.adaptive_constant = pde_model.adaptive_constant * (1.0 - pde_model.beta) + pde_model.beta * adaptive_constant_tmp
                pde_model.adaptive_constant = max(min(pde_model.adaptive_constant,100),1)
                # print(f"==>> pde_model.adaptive_constant: {pde_model.adaptive_constant}")

        closure.loss_bc = loss_bc
        closure.loss_f  = loss_f
        return loss   
    
    #==========================================================================================================================
    flag = False if os.path.isfile(save_dir+LossFile) else True
    
    with open(save_dir+LossFile, mode=mode) as f:
        if mode =='a':
            if flag:
                if data.phi_test is not None  and data.p_test is not None:
                    f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi Error_p\n")
                elif data.phi_test is not None and data.p_test is  None:
                    f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi\n")
                else:
                    f.write(f"Iter Loss Loss_BC Loss_PDE\n")
            else:
                print(f"{LossFile} exist !, Continue to write")
        else:
            if data.phi_test is not None  and data.p_test is not None:
                f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi Error_p\n")
            elif data.phi_test is not None and data.p_test is  None:
                f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi\n")
            else:
                f.write(f"Iter Loss Loss_BC Loss_PDE\n")
            

        for it in tqdm(range(n_step)):
            # Call dataloader
            if train_on_batch:
                for step, (X_train, X_bc) in enumerate(zip(data.data_collocation, data.data_boundary)):
                    # print(f'step={step}, X_train={X_train.shape}, X_bc = {X_bc.shape}')
                    loss = pde_model.optimizer.step(closure=closure)
                    if torch.cuda.is_available():
                        del X_train
                        del X_bc
                        torch.cuda.empty_cache()
            else: # train on full data set
                X_train, X_bc = data.X_train, data.X_bc
                X_interface = data.X_interface
                loss = pde_model.optimizer.step(closure=closure)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            if pde_model.scheduler is not None:
                pde_model.scheduler.step()

            loss_bc = closure.loss_bc
            loss_f  = closure.loss_f
            # Test the model and log information 
            if pde_model.iter % log_every == 0 or pde_model.iter==1:
                if data.phi_test is not None:
                    phi_REL_L2,_,_,_,mass_phi_pred,mass_phi_test  = pde_model.get_phi_test(data.X_test,data.phi_test,normalization,ngroup)
                    if data.p_test is not None:
                        p_REL_L2 = pde_model.get_currents_test(data.Xc_test,data.p_test,normalization,mass_phi_pred,mass_phi_test,ngroup)[0]
                        f.write(f"{pde_model.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f} {phi_REL_L2.item():<12.5f} {p_REL_L2.item():<12.5f}\n")
                    else:
                        f.write(f"{pde_model.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f} {phi_REL_L2.item():<12.5f} \n")


                    if verbose ==1:
                        print(f"{pde_model.iter:6d}: Loss = {loss.item():>12.5f}; Error_phi = {phi_REL_L2.item():<12.5f}")
                    # save model according to test set
                    if phi_REL_L2.item() < pde_model.best_test:
                        torch.save(pde_model.model.state_dict(),save_dir+ "model_test.pt")
                        pde_model.best_test = phi_REL_L2.item()
                else:
                    f.write(f"{pde_model.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f}\n")
                    if verbose==1:
                        print(f"{pde_model.iter:6d}: Loss = {loss.item():>12.5f}; ")
            
            # Save the model if it improves according to the loss function
            flag_eigen = True # for eigenvalue problem, we always save model
            if loss.item() < pde_model.best_train or flag_eigen:
                torch.save(pde_model.model.state_dict(),save_dir+ "model.pt")
                pde_model.best_train = loss.item()

            
            

            # Modify data
            if resampling_every and pde_model.iter % resampling_every == 0 and pde_model.iter >20000 and pde_model.iter <=100000:
                print('resampling!')
                # data.update_random_collocation_points(random_seed=pde_model.iter )
                residu = pde_model.residual_PDE(data.X_train)
                residu = torch.pow(residu,2)
                residu = torch.sum(residu,1,keepdim=True)
                if pde_model.iter % 5000 ==0:
                    visualization.viewCrossSection(data.X_train,S=residu,pathFile=save_dir+'residu_train_'+str(pde_model.iter)+'.png')
                data.X_train = pde_model.adaptive_sampling(data.X_train,method='RAD')
                if pde_model.iter % 5000 ==0:
                    visualization.viewCrossSection(data.X_train,S=None,pathFile=save_dir+'X_train_'+str(pde_model.iter)+'.png')

            # Update source for the eigenvalue problem:
            if pde_model.do_outer and pde_model.iter % pde_model.N_inner == 0:
                pde_model.update_source(data.X_train)


            pde_model.iter += 1
            if pde_model.stopping:
                print('Finish!')
                break
            # reset learning rate
            # if pde_model.iter==100000:
            #     for id in range(len(pde_model.optimizer.param_groups)):
            #         pde_model.optimizer.param_groups[id]['lr'] = 0.001
            #         print('Recover learning rate !',pde_model.optimizer.param_groups[id]['lr'] )

    tout=time.time()
    print("Elapsed: ",tout-tin," seconds")












