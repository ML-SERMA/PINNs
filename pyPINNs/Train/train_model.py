import torch
import time
from tqdm import tqdm
import os
from ..Tools.visualization import visualization
from ..Tools.logger import Logger



def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def evaluate_model(pde_model, data, normalization):
    if data.phi_test is not None:
        errors = []
        phi_REL_L2_global,phi_REL_L2, _, _, _, mass_phi_pred, mass_phi_test = pde_model.get_phi_test(data.X_test, data.phi_test, normalization)
        errors.append(phi_REL_L2_global)
        errors.extend(phi_REL_L2)
        if data.p_test is not None:
            p_REL_L2_global,p_REL_L2 = pde_model.get_currents_test(data.Xc_test, data.p_test, normalization, mass_phi_pred, mass_phi_test)[0:2]
            errors.append(p_REL_L2_global)
            errors.extend(p_REL_L2)
        return errors
    else:
        return None

# class EarlyStopping:
#     def __init__(self):
#         self.stop = False

#     def should_stop(self, pde_model):
#         return getattr(pde_model, "stopping", False)


def train(
    pde_model,
    data,
    n_step=100,
    verbose=0,
    log_every=100,
    LossFile='Loss.dat',
    save_dir='',
    adaptive_loss=None,
    attention_loss=None,
    model_regularizer=None,
    physics_regularizer=None,
    resampler = None,
    mode='w+',
    **kwargs
):
    train_on_batch = kwargs.get('train_on_batch', False)
    normalization = kwargs.get('normalization', False)
    # num_groups = kwargs.get('num_groups', None)
    isLossData = kwargs.get('isLossData', False)
    save_model_always = kwargs.get('save_model_always', True)


    if isLossData:
        print('Generate exp points !')
        data.generate_exp_points()
    # Prepare headers for logger based on available data
    header_loss = ['Iter','Loss', 'Loss_BC', 'Loss_PDE']
    header_errors = []
    # if data.phi_test is not None:
    #     header_errors.append('Error_phi')
    # if data.p_test is not None:
    #     header_errors.append('Error_p')

    if data.phi_test is not None:
        # Flux: global + per-group
        header_errors.append('Error_phi_global')
        for g in range(len(data.phi_test)):  # number of groups
            header_errors.append(f'Error_phi_g{g}')

    if getattr(data, 'p_test', None) is not None:
        # Currents: global + per-group
        header_errors.append('Error_p_global')
        for g in range(len(data.p_test)):  # number of groups
            header_errors.append(f'Error_p_g{g}')

    logger = Logger(
        filepath=LossFile,
        mode=mode,
        save_dir=save_dir,
        header_main=header_loss,
        header_extra=header_errors,
        verbose=verbose
    )

    tin = time.time()
    if verbose == 3:
        for key, value in kwargs.items():
            print(f"{key} == {value}")
        print(f'n_step = {n_step}, log = {log_every}, LossFile = {LossFile}, save_dir = {save_dir}')

    def closure():
        pde_model.optimizer.zero_grad()

        if adaptive_loss is not None:
            loss_f = pde_model.loss_PDE(X_train)
            loss_bc = pde_model.loss_BC(X_bc)
            loss_physics = adaptive_loss([loss_f, loss_bc], model=pde_model.model)
        elif attention_loss is not None:
            loss_f = attention_loss(pde_model.residual_PDE(X_train))
            loss_bc = pde_model.loss_BC(X_bc)
            loss_physics = loss_f + loss_bc
        else:
            loss_f = pde_model.loss_PDE(X_train)
            loss_bc = pde_model.loss_BC(X_bc)
            loss_physics = loss_f + loss_bc

        # Regularization losses
        loss = loss_physics

        if model_regularizer is not None:
            loss += model_regularizer.compute(pde_model.model)

        if physics_regularizer is not None:
            loss_reg= physics_regularizer.compute(
                pde_model, X_train=X_train, X_interface=X_interface, data=data
            )
            loss = loss + loss_reg

        loss.backward(retain_graph=True)


        closure.loss_f = loss_f
        closure.loss_bc = loss_bc
        return loss
    
    # torch.autograd.set_detect_anomaly(True)  # Enable for debugging (optional)

    for it in tqdm(range(n_step)):
        # Call dataloader
        if train_on_batch:
            for _, (X_train, X_bc) in enumerate(zip(data.data_collocation, data.data_boundary)):

                loss = pde_model.optimizer.step(closure=closure)
                # loss = pde_model.optimizer.step(lambda: closure(X_train, X_bc, X_interface))
                clear_cuda()
                
        else: # train on full data set
            X_train, X_bc = data.X_train, data.X_bc
            X_interface = data.X_interface

            # # Update attention MLP 
            # if attention_loss is not None:
            #     residuals = pde_model.residual_PDE(X_train)
            #     attention_loss.update_attention(residuals)

            loss = pde_model.optimizer.step(closure=closure)
            # loss = pde_model.optimizer.step(lambda: closure(X_train, X_bc, X_interface))


            # Update attention MLP 
            # if attention_loss is not None:
            #     residuals = pde_model.residual_PDE(X_train).detach()
            #     attention_loss.update_attention(residuals)


            clear_cuda()

        if pde_model.scheduler is not None:
            pde_model.scheduler.step()
            # print learning rate 
            # if it % pde_model.scheduler.step_size == 0:
            #     print(f"iter={it}, lr={pde_model.scheduler.get_lr()[0]:.3e}")


        if pde_model.iter % log_every == 0 or pde_model.iter == 1:
            losses = [pde_model.iter,loss.item(), closure.loss_bc.item(), closure.loss_f.item()]
            output_errors = evaluate_model(pde_model, data, normalization)
            if output_errors is not None:
                errors =  [err.item()  for err in output_errors]
                if  errors[0] < pde_model.best_test: # the first important error: flux error
                    torch.save(pde_model.model.state_dict(), os.path.join(save_dir, "model_test.pt"))
                    pde_model.best_test = errors[0]
            else:
                errors = None
            
            # Log using Logger
            logger.log(losses, errors if errors else None)

        # Save best model
        if loss.item() < pde_model.best_train or save_model_always:
            torch.save(pde_model.model.state_dict(), os.path.join(save_dir, "model.pt"))
            pde_model.best_train = loss.item()

        # Resampling
        if resampler is not None and resampler.should_resample(pde_model.iter):
            resampler.apply(pde_model.iter)
            pde_model.get_cross_sections(data.X_train)



        if getattr(pde_model, "do_outer", False) and getattr(pde_model, "N_inner", 0) > 0:
            if pde_model.iter % pde_model.N_inner == 0:
                if hasattr(pde_model, "update_source"):
                    pde_model.update_source(data.X_train)
                    # pde_model.update_Sgr_train(data.X_train)
                    # attention_loss.log_weights()
    
        pde_model.iter += 1
        if getattr(pde_model, "stopping", False):
            print("Finish!")
            break

    tout=time.time()
    print("Elapsed: ",tout-tin," seconds")














##========================================================================
##========================================================================
##=========================================================================
# def compute_l1_loss(w):
#     return torch.abs(w).sum()

# def compute_l2_loss(w):
#     return torch.square(w).sum()

# def train_old_version(pde_model,data,n_step=100,verbose=0, log_every=100, LossFile='Loss.dat',save_dir = '',**kwargs):

#     train_on_batch = kwargs.get('train_on_batch',False)
#     learning_rate_annealing = kwargs.get('learning_rate_annealing',False)
#     weight_ridge =  kwargs.get('weight_ridge',0)
#     weight_lasso =  kwargs.get('weight_lasso',0)
#     weight_Sobolev = kwargs.get('weight_Sobolve',0)
#     weight_regularization = kwargs.get('weight_regularization',0)
#     weight_interface = kwargs.get('weight_interface',0)
#     resampling_every = kwargs.get('resampling_every',0)
#     normalization = kwargs.get('normalization',False)
#     ngroup        = kwargs.get('ngroup',None)
#     mode = kwargs.get('mode','w+')


#     isLossData = kwargs.get('isLossData',False)
#     if isLossData:
#         print('Generate exp points !')
#         data.generate_exp_points()


#     # torch.autograd.set_detect_anomaly(True)
#     tin=time.time()
#     if verbose ==3 :
#         for key, value in kwargs.items():
#             print("%s == %s" % (key, value))
#         print(f'n_step = {n_step} , log = {log_every}, lossFile = {LossFile}, savedir = {save_dir}')

#     # LR annealing
#     pde_model.beta = 0.8 
#     pde_model.adaptive_constant = 1.0
    
#     # mymodel = copy.deepcopy(pde_model)

#     def closure():
#         pde_model.optimizer.zero_grad()
#         loss_bc  = pde_model.loss_BC(X_bc)
#         loss_f = pde_model.loss_PDE(X_train)
#         loss = pde_model.adaptive_constant*(loss_bc) + loss_f

#         #============== Loss data =========================
#         if isLossData:
#             loss_data = pde_model.loss_EXP(data.X_exp,data.phi_exp)
#             loss = loss + loss_data
        
#         # Regularization
#         if weight_lasso or weight_ridge:
#             parameters = []
#             for parameter in pde_model.model.parameters():
#                 parameters.append(parameter.view(-1))
#         if weight_lasso:
#             l1 = weight_lasso * compute_l1_loss(torch.cat(parameters))
#             loss += l1
#         if weight_ridge:
#             l2 = weight_ridge * compute_l2_loss(torch.cat(parameters))
#             loss += l2
#         if weight_Sobolev:
#             lS = weight_Sobolev * pde_model.Sobolev_norm(X_train)
#             loss +=lS
#         if weight_regularization:
#             lReg = weight_regularization* pde_model.mass_regularization(X_train)
#             loss = loss + lReg
#         if weight_interface:
#             loss_jump = weight_interface*pde_model.interface_jump(X_interface)
#             loss = loss + loss_jump

        
#         loss.backward(retain_graph=True)

#         ###############################################################################################
#         ### GRADIENT CLIPPING
#         # torch.nn.utils.clip_grad_norm_(pde_model.model.parameters(), 10.) # improves training stability
#         ###############################################################################################

        
        
#         '# Evaluation of adaptive_constant'
#         if learning_rate_annealing and pde_model.iter % 10 == 0 and pde_model.iter > 0:
#             with torch.no_grad():   # To avoid OOM
#                 grad_f_list  = (torch.autograd.grad(loss_f , pde_model.model.linears[0].weight, retain_graph=True, create_graph=True)[0]).flatten()
#                 grad_bc_list = (torch.autograd.grad(loss_bc, pde_model.model.linears[0].weight, retain_graph=True, create_graph=True)[0]).flatten()

#                 for i in range(len(pde_model.model.layers)-2):
#                     grad_f_list  = torch.cat((grad_f_list , (torch.autograd.grad(loss_f , pde_model.model.linears[i+1].weight, retain_graph=True, create_graph=True)[0]).flatten() ))
#                     grad_bc_list = torch.cat((grad_bc_list, (torch.autograd.grad(loss_bc, pde_model.model.linears[i+1].weight, retain_graph=True, create_graph=True)[0]).flatten() ))
                    
#                 print(f"==>> grad_f_list: {grad_f_list}") 
#                 print(f"==>> grad_bc_list: {grad_bc_list}")
#                 max_grad_res           = torch.max(torch.abs(grad_f_list))
#                 print(f"==>> max_grad_res: {max_grad_res}")
#                 mean_grad_bcs          = torch.mean(torch.abs(grad_bc_list)) 
#                 print(f"==>> mean_grad_bcs: {mean_grad_bcs}")
#                 # mean_grad_res          = torch.mean(torch.abs(grad_f_list))

#                 adaptive_constant_tmp  = max_grad_res / mean_grad_bcs
#                 pde_model.adaptive_constant = pde_model.adaptive_constant * (1.0 - pde_model.beta) + pde_model.beta * adaptive_constant_tmp
#                 pde_model.adaptive_constant = max(min(pde_model.adaptive_constant,100),1)
#                 print(f"==>> pde_model.adaptive_constant: {pde_model.adaptive_constant}")

#         closure.loss_bc = loss_bc
#         closure.loss_f  = loss_f
#         return loss   
    
#     #==========================================================================================================================
#     flag = False if os.path.isfile(save_dir+LossFile) else True
    
#     with open(save_dir+LossFile, mode=mode) as f:
#         if mode =='a':
#             if flag:
#                 if data.phi_test is not None  and data.p_test is not None:
#                     f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi Error_p\n")
#                 elif data.phi_test is not None and data.p_test is  None:
#                     f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi\n")
#                 else:
#                     f.write(f"Iter Loss Loss_BC Loss_PDE\n")
#             else:
#                 print(f"{LossFile} exist !, Continue to write")
#         else:
#             if data.phi_test is not None  and data.p_test is not None:
#                 f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi Error_p\n")
#             elif data.phi_test is not None and data.p_test is  None:
#                 f.write(f"Iter Loss Loss_BC Loss_PDE Error_phi\n")
#             else:
#                 f.write(f"Iter Loss Loss_BC Loss_PDE\n")
            

#         for it in tqdm(range(n_step)):
#             # Call dataloader
#             if train_on_batch:
#                 for step, (X_train, X_bc) in enumerate(zip(data.data_collocation, data.data_boundary)):
#                     # print(f'step={step}, X_train={X_train.shape}, X_bc = {X_bc.shape}')
#                     loss = pde_model.optimizer.step(closure=closure)
#                     if torch.cuda.is_available():
#                         del X_train
#                         del X_bc
#                         torch.cuda.empty_cache()
#             else: # train on full data set
#                 X_train, X_bc = data.X_train, data.X_bc
#                 X_interface = data.X_interface
#                 loss = pde_model.optimizer.step(closure=closure)
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()


#             if pde_model.scheduler is not None:
#                 pde_model.scheduler.step()

#             loss_bc = closure.loss_bc
#             loss_f  = closure.loss_f
#             # Test the model and log information 
#             if pde_model.iter % log_every == 0 or pde_model.iter==1:
#                 if data.phi_test is not None:
#                     phi_REL_L2,_,_,_,mass_phi_pred,mass_phi_test  = pde_model.get_phi_test(data.X_test,data.phi_test,normalization,ngroup)
#                     if data.p_test is not None:
#                         p_REL_L2 = pde_model.get_currents_test(data.Xc_test,data.p_test,normalization,mass_phi_pred,mass_phi_test,ngroup)[0]
#                         f.write(f"{pde_model.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f} {phi_REL_L2.item():<12.5f} {p_REL_L2.item():<12.5f}\n")
#                     else:
#                         f.write(f"{pde_model.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f} {phi_REL_L2.item():<12.5f} \n")


#                     if verbose ==1:
#                         print(f"{pde_model.iter:6d}: Loss = {loss.item():>12.5f}; Error_phi = {phi_REL_L2.item():<12.5f}")
#                     # save model according to test set
#                     if phi_REL_L2.item() < pde_model.best_test:
#                         torch.save(pde_model.model.state_dict(),save_dir+ "model_test.pt")
#                         pde_model.best_test = phi_REL_L2.item()
#                 else:
#                     f.write(f"{pde_model.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f}\n")
#                     if verbose==1:
#                         print(f"{pde_model.iter:6d}: Loss = {loss.item():>12.5f}; ")
            
#             # Save the model if it improves according to the loss function
#             flag_eigen = True # for eigenvalue problem, we always save model
#             if loss.item() < pde_model.best_train or flag_eigen:
#                 torch.save(pde_model.model.state_dict(),save_dir+ "model.pt")
#                 pde_model.best_train = loss.item()

            
            

#             # Modify data
#             if resampling_every and pde_model.iter % resampling_every == 0 and pde_model.iter >20000 and pde_model.iter <=100000:
#                 print('resampling!')
#                 # data.update_random_collocation_points(random_seed=pde_model.iter )
#                 residu = pde_model.residual_PDE(data.X_train)
#                 residu = torch.pow(residu,2)
#                 residu = torch.sum(residu,1,keepdim=True)
#                 if pde_model.iter % 5000 ==0:
#                     visualization.viewCrossSection(data.X_train,S=residu,pathFile=save_dir+'residu_train_'+str(pde_model.iter)+'.png')
#                 data.X_train = pde_model.adaptive_sampling(data.X_train,method='RAD')
#                 if pde_model.iter % 5000 ==0:
#                     visualization.viewCrossSection(data.X_train,S=None,pathFile=save_dir+'X_train_'+str(pde_model.iter)+'.png')

#             # Update source for the eigenvalue problem:
#             if pde_model.do_outer and pde_model.iter % pde_model.N_inner == 0:
#                 pde_model.update_source(data.X_train)


#             pde_model.iter += 1
#             if pde_model.stopping:
#                 print('Finish!')
#                 break

#     tout=time.time()
#     print("Elapsed: ",tout-tin," seconds")












