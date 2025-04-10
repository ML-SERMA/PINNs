
import torch
import torch.nn as nn


class multiGroupLoss(nn.Module):
    def __init__(self, num_groups, method="weighted", init_weights=None,lr_weights=None):
        """
        Args:
            num_groups (int): Number of neutron energy groups.
            method (str): Loss balancing method: "weighted", "log", "normalized", or "uncertainty".
            init_weights (list or None): Initial loss weights for weighted loss.
        """
        super(multiGroupLoss, self).__init__()
        self.num_groups = num_groups
        self.method = method
        if method == "weighted" and init_weights is  None:
            self.weights = torch.ones(num_groups, dtype=torch.float32)

        if method == "uncertainty":
            # Trainable uncertainty parameters
            self.sigma = nn.Parameter(torch.ones(num_groups))
        
        if method == "weighted" and init_weights is not None:
            self.weights = torch.tensor(init_weights, dtype=torch.float32)
        
        if method == "adaptive":
            self.log_weights = torch.nn.Parameter(torch.ones(num_groups))  # Log-weights (λ_i)
            if lr_weights is None:
                self.lr_weights=0.001
            else:
                self.lr_weights = lr_weights  # Learning rate for gradient ascent on weights

    def forward(self, loss_terms, model=None):
        """
        Args:
            loss_terms (list of Tensors): List of individual loss terms for each group.
        Returns:
            torch.Tensor: Total loss.
        """
        loss_terms = torch.stack(loss_terms)  # Shape: (num_groups,)
        
        if self.method == "weighted":
            return torch.sum(self.weights.to(loss_terms.device) * loss_terms)

        elif self.method == "log":
            return torch.sum(torch.log(1 + loss_terms))

        elif self.method == "normalized":
            normalized_loss = loss_terms / (loss_terms.max() + 1e-8)
            return torch.sum(normalized_loss)

        elif self.method == "uncertainty":
            return torch.sum((1.0 / (2 * self.sigma**2)) * loss_terms) + torch.sum(torch.log(self.sigma))
        
        elif self.method == "ntk":
            if model is None:
                raise ValueError("NTK-based weighting requires the model to compute gradients.")

            # Compute NTK-based loss weights
            grads = []
            for loss in loss_terms:
                grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
                grad_norm = torch.cat([g.view(-1) for g in grad if g is not None]).norm()  # Compute norm
                grads.append(grad_norm + 1e-8)  # Add small epsilon to avoid division by zero

            grads = torch.stack(grads)  # Shape: (num_groups,)

            # Compute NTK-based weights
            inv_grad_norms = 1.0 / grads
            weights = inv_grad_norms / torch.sum(inv_grad_norms)  # Normalize to sum to 1

            return torch.sum(weights * loss_terms)
        
        elif self.method == "adaptive":

            self.weights = torch.softmax(self.log_weights, dim=0)  # Convert to valid weights
            # print(f"==>> weights: {self.weights}")

            return torch.sum(self.weights.to(loss_terms.device) * loss_terms)

        else:
            raise ValueError("Invalid loss balancing method!")
        
    def update_weights(self, total_loss):
        """ Perform gradient ascent on the weights """
        # total_loss = self.forward(losses)

        # Compute gradient w.r.t log-weights
        grads = torch.autograd.grad(total_loss, self.log_weights, retain_graph=True)[0]

        # Update log-weights with gradient ascent
        with torch.no_grad():
            self.log_weights += self.lr_weights * grads
            self.log_weights -= self.log_weights.mean()  # Normalize to prevent drift