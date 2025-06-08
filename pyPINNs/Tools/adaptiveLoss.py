import torch
import torch.nn as nn


class adaptiveLoss(nn.Module):
    def __init__(self, method="weighted", init_weights=None, lr_weights=1e-3, temperature=1.0, beta=0.8, update_interval=10):
        super().__init__()
        self.method = method.lower()
        self.lr_weights = lr_weights
        self.temperature = temperature
        self.beta = beta
        self.init_weights = init_weights
        self.initialized = False
        self.iter = 0
        self.update_interval = update_interval


    def _initialize(self, num_terms, device):
        self.num_terms = num_terms

        if self.method in ["weighted", "grad_annealing"]:
            if self.init_weights is not None:
                self.weights = torch.tensor(self.init_weights, dtype=torch.float32, device=device)
            else:
                self.weights = torch.ones(num_terms, dtype=torch.float32, device=device)
        elif self.method == "adaptive":
            self.log_weights = nn.Parameter(torch.zeros(num_terms, device=device))
        self.last_weights = torch.ones(num_terms, device=device)
        self.initialized = True

    def forward(self, loss_terms, model=None):
        loss_terms = torch.stack(loss_terms)
        if not self.initialized:
            self._initialize(len(loss_terms), loss_terms.device)

        if self.method == "grad_annealing" and self.iter % self.update_interval == 0 and self.iter > 0:
            if model is None:
                raise ValueError("Model must be provided for grad_annealing ! since we have to compute grad of the network.")
            self._update_grad_annealing_weights(loss_terms, model)

        if self.method == "adaptive" and self.iter % self.update_interval == 0 and self.iter > 0:
            self._update_adaptive_weights(loss_terms)

        self.iter += 1
        weights = self._compute_weights()
        self.last_weights = weights.detach()
        return torch.sum(weights * loss_terms)

    def _compute_weights(self):

        if self.method == "weighted":
            return self.weights / self.weights.sum()
        elif self.method == "adaptive":
            return torch.softmax(self.log_weights / self.temperature, dim=0)
        elif self.method == "grad_annealing":
            return self.weights 
        else:
            raise NotImplementedError(f"Method '{self.method}' not implemented.")

    def _update_grad_annealing_weights(self, loss_terms, model):

        loss_f = loss_terms[0]
        loss_bc = loss_terms[1]

        
        with torch.no_grad():
            grads_f = self._get_full_gradient(loss_f, model)
            print(f"==>> grads_f: {grads_f}")
            grads_bc = self._get_full_gradient(loss_bc, model)
            print(f"==>> grads_bc: {grads_bc}")
            max_grad_f = torch.max(torch.abs(grads_f))
            print(f"==>> max_grad_f: {max_grad_f}")
            mean_grad_bc = torch.mean(torch.abs(grads_bc))
            print(f"==>> mean_grad_bc: {mean_grad_bc}")

            if mean_grad_bc > 1e-12:
                adaptive_constant_tmp = max_grad_f / mean_grad_bc
                new_weight = self.weights[1] * (1.0 - self.beta) + self.beta * adaptive_constant_tmp
                self.weights[1] = torch.clamp(new_weight, min=1.0, max=100.0)
                self.weights[0] = 1.0
                print(f"==>> weights: {self.weights}")
                

    def _update_adaptive_weights(self, loss_terms):
        total_loss = torch.sum(torch.softmax(self.log_weights / self.temperature, dim=0) * loss_terms)
        grads = torch.autograd.grad(total_loss, self.log_weights, retain_graph=True)[0]
        with torch.no_grad():
            self.log_weights += self.lr_weights * grads
            self.log_weights -= self.log_weights.mean()  # Normalize

    def _get_full_gradient(self, loss, model):
        grads = []
        for layer in model.linears:
            grad = torch.autograd.grad(loss, layer.weight, retain_graph=True, create_graph=True)[0]
            grads.append(grad.flatten())
        return torch.cat(grads)

    def get_weights(self):
        return self.last_weights.cpu().clone()

    def get_log_weights(self):
        if self.method == "adaptive":
            return self.log_weights.detach().cpu().clone()
        return None








