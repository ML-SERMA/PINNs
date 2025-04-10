import torch
import numpy as np

class operator:
    @staticmethod
    def grad(model_out, *derivative_variable):
        '''Computes the gradient of a network with respect to the given variable.
        Parameters
        ----------
        model_out : torch.tensor
            The (scalar) output tensor of the neural network
        derivative_variable : torch.tensor
            The input tensor of the variables in which respect the derivatives have to
            be computed
        Returns
        ----------
        torch.tensor
            A Tensor, where every row contains the values of the the first
            derivatives (gradient) w.r.t the row of the input variable.
        '''
        grad = []
        for vari in derivative_variable:
            new_grad = torch.autograd.grad(model_out.sum(), vari,create_graph=True, allow_unused=True)[0]
            grad.append(new_grad)
        return torch.column_stack(grad)
    @staticmethod
    def div(model_out, *derivative_variable):
        '''Computes the divergence of a network with respect to the given variable.
        Only for vector valued inputs, for matices use the function matrix_div.
        Parameters
        ----------
        model_out : torch.tensor
            The output tensor of the neural network
        derivative_variable : torch.tensor
            The input tensor of the variables in which respect the derivatives have to
            be computed. Have to be in a consistent ordering, if for example the output
            is u = (u_x, u_y) than the variables has to passed in the order (x, y)
        Returns
        ----------
        torch.tensor
            A Tensor, where every row contains the values of the divergence
            of the model w.r.t the row of the input variable.
        '''
        divergence = torch.zeros((*derivative_variable[0].shape[:-1], 1),
                                device=derivative_variable[0].device)
        var_dim = 0
        for vari in derivative_variable:
            for i in range(vari.shape[-1]):
                Du = torch.autograd.grad(model_out.narrow(-1, var_dim + i, 1).sum(),
                                        vari, create_graph=True, allow_unused=True)[0]
                divergence = divergence + Du.narrow(-1, i, 1)
            var_dim += i + 1
        return divergence