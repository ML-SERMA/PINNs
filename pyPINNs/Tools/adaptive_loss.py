import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class AdaptiveLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.iter = 0
        self.weight_log = []

    @abstractmethod
    def forward(self, losses, model=None):
        pass
    @staticmethod
    def _normalize_weights(weights, method="softmax"):
        if method == "softmax":
            return torch.softmax(weights, dim=0)
        elif method == "sum":
            return weights / weights.sum()
        else:
            return weights

    def record_weights(self, weights):
        self.weight_log.append(weights.detach().cpu().numpy())

    def plot_weights(self, labels=None, title="Loss Weights Over Time"):
        if not self.weight_log:
            print("No weights logged.")
            return
        weight_tensor = torch.tensor(self.weight_log)
        for i in range(weight_tensor.shape[1]):
            label = labels[i] if labels else f"Term {i}"
            plt.plot(weight_tensor[:, i], label=label)
        plt.xlabel("Iterations")
        plt.ylabel("Weight")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class WeightedLoss(AdaptiveLoss):
    def __init__(self, init_weights):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32), requires_grad=False)

    def forward(self, losses, model=None):
        loss_tensor = torch.stack(list(losses.values()) if isinstance(losses, dict) else losses)
        weights = self._normalize_weights(self.weights,method='sum')
        self.record_weights(weights)
        self.iter += 1
        return torch.sum(weights * loss_tensor)


class AdaptiveSoftmaxLoss(AdaptiveLoss):
    def __init__(self, n_terms, lr=1e-2, temperature=1.0):
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(n_terms))
        self.lr = lr
        self.temperature = temperature

    def forward(self, losses, model=None):
        loss_tensor = torch.stack(list(losses.values()) if isinstance(losses, dict) else losses)
        weights = torch.softmax(self.weight_logits / self.temperature, dim=0)
        total_loss = torch.sum(weights * loss_tensor)

        grads = torch.autograd.grad(total_loss, self.weight_logits, retain_graph=True)[0]
        with torch.no_grad():
            self.weight_logits += self.lr * grads
            self.weight_logits -= self.weight_logits.mean()  # normalize

        self.record_weights(weights)
        self.iter += 1
        return total_loss


class GradAnnealingLoss(AdaptiveLoss):
    def __init__(self, beta=0.9, update_interval=10):
        super().__init__()
        self.beta = beta
        self.update_interval = update_interval
        self.weights = None

    def forward(self, losses, model):
        loss_tensor = torch.stack(list(losses.values()) if isinstance(losses, dict) else losses)
        if self.weights is None:
            self.weights = torch.ones_like(loss_tensor)

        if self.iter % self.update_interval == 0 and self.iter > 0:
            grads = [self._grad_norm(loss, model) for loss in loss_tensor]
            grads = torch.tensor(grads, device=loss_tensor.device)
            self.weights = self.beta * self.weights + (1 - self.beta) * grads

        weights = self._normalize_weights(self.weights, method='sum')
        self.record_weights(weights)
        self.iter += 1
        return torch.sum(weights * loss_tensor)

    def _grad_norm(self, loss, model):
        grads = []
        for p in model.parameters():
            if p.requires_grad:
                g = torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0]
                if g is not None:
                    grads.append(g.view(-1))
        if grads:
            return torch.cat(grads).norm().item()
        return 0.0


class ListResidualAttention(nn.Module):
    def __init__(self,  lr=1e-2, temperature=1.0):
        """
        Pointwise residual attention weighting.
        - n_terms: number of residual loss terms
        - lr: learning rate for attention weights
        - temperature: softmax temperature
        """
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        self.attention_logits = []  # List of tensors [N_i] for each residual
        self.initialized = False
        self.iter = 0  # <- ADD THIS LINE
 

    def forward(self, residuals, model=None):
        """
        residuals: list of residual tensors, one for each PDE term
                  Each tensor has shape (N_i,) for term i
        """
        if isinstance(residuals, dict):
            residuals = list(residuals.values())

        if not self.initialized:
            self.attention_logits = [
                nn.Parameter(torch.zeros_like(r)) for r in residuals
            ]
            self.initialized = True

        total_loss = 0.0
        pointwise_weights = []
        
        
        for residual, logits in zip(residuals, self.attention_logits):
            weights = torch.softmax(logits / self.temperature, dim=0)
            loss_term = torch.sum(weights * residual.pow(2))

            grads = torch.autograd.grad(loss_term, logits, retain_graph=True)[0]
            with torch.no_grad():
                logits += self.lr * grads
                logits -= logits.mean()  # normalize

            pointwise_weights.append(weights.detach().cpu())

            total_loss += loss_term


        # self.record_weights(torch.cat([w.mean().unsqueeze(0) for w in pointwise_weights]))
        self.iter += 1
        return total_loss
    


def get_adaptive_loss(method: str, **kwargs):
    method = method.lower()
    if method == "weighted":
        return WeightedLoss(**kwargs)
    elif method == "adaptive":
        return AdaptiveSoftmaxLoss(**kwargs)
    elif method == "grad_annealing":
        return GradAnnealingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown adaptive loss method: '{method}'")
    




##========================================================================================================================================




class ResidualAttention(nn.Module):
    def __init__(self, lr=1e-3, temperature=1.0, mode="point"):
        """
        Residual attention for multigroup neutron diffusion.
        Supports 3D residuals: (N, G, T) = (points, groups, terms)
        - lr: learning rate for attention weights
        - temperature: softmax temperature
        - mode: one of ['point', 'group', 'term', 'all']
        """
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        self.mode = mode
        self.attention_logits = None  # tensor of shape (N, G, T)
        self.initialized = False
        self.iter = 0

    def forward(self, residuals):
        """
        Args:
            residuals: tensor of shape (N, G, T), squared residuals per point/group/term
        """
        if isinstance(residuals, dict):
            residuals = torch.stack(list(residuals.values()), dim=-1)  # (N, G, T)

        if residuals.dim() == 2:  # (N, G)
            residuals = residuals.unsqueeze(2)  # -> (N, G, 1)

        if residuals.dim() != 3:
            raise ValueError(f"Expected residuals of shape (N, G, T), got {residuals.shape}")

        if not self.initialized:
            self.attention_logits = nn.Parameter(torch.zeros_like(residuals), requires_grad=True)
            self.initialized = True

        weights = self.normalize_logits(self.attention_logits, self.temperature, self.mode)
        print(f"==>> Iter: {self.iter}, weights: {weights}")
        weighted_loss = torch.sum(weights * residuals.pow(2))

        # Backprop for attention logits
        grads = torch.autograd.grad(weighted_loss, self.attention_logits, retain_graph=True)[0]
        print("Gradient norm (should not be zero):", grads.norm())
        with torch.no_grad():
            self.attention_logits += self.lr * grads
            # self.attention_logits -= self.attention_logits.mean()  # optional normalization

        self.iter += 1
        return weighted_loss

    @staticmethod
    def normalize_logits(logits, temperature=1.0, mode="point"):
        """
        Normalize logits into softmax weights.
        Args:
            logits: Tensor (N, G, T)
            mode: 'point' → softmax over T
                  'group' → softmax over G
                  'term' → softmax over N
                  'all' → softmax over all (flattened)
        Returns:
            Softmax weights of shape (N, G, T)
        """
        logits = logits / temperature
        if mode == "point":
            return torch.softmax(logits, dim=2)  # over T
        elif mode == "group":
            return torch.softmax(logits, dim=1)  # over G
        elif mode == "term":
            return torch.softmax(logits, dim=0)  # over N
        elif mode == "all":
            # flat_logits = logits.reshape(-1)  # keeps computation graph
            # flat_weights = torch.softmax(flat_logits, dim=0)
            # return flat_weights.reshape_as(logits)

            # flat = logits.view(-1)
            # return torch.softmax(flat, dim=0).view_as(logits)

            # logits_ = logits.clone()
            # logits_ = logits_.view(-1)  # same shape, but preserve gradient
            # weights = torch.softmax(logits_, dim=0)
            # return weights.view_as(logits)

            max_logits = logits.max()
            exp_logits = torch.exp(logits - max_logits)  # for stability
            sum_exp = exp_logits.sum()
            weights = exp_logits / sum_exp
            return weights
        else:
            raise ValueError(f"Unknown mode '{mode}' for normalization")
        




class ResidualAttentionNet(nn.Module):
    def __init__(self, hidden_dim=16, lr=1e-3):
        """
        Residual attention using neural network (not softmax).
        - hidden_dim: hidden size for attention MLP
        - lr: learning rate for attention net
        """
        super().__init__()
        self.lr = lr
        self.attention_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures output is positive
        )
        self.iter = 0

    def forward(self, residuals):
        """
        Args:
            residuals: tensor of shape (N, G, T)
        Returns:
            Weighted loss scalar
        """
        if isinstance(residuals, dict):
            residuals = torch.stack(list(residuals.values()), dim=-1)  # (N, G, T)

        if residuals.dim() == 2:  # (N, G)
            residuals = residuals.unsqueeze(-1)  # -> (N, G, 1)

        x = residuals.detach().pow(2).unsqueeze(-1)  # (N, G, T, 1) for MLP
        weights = self.attention_net(x).squeeze(-1)  # (N, G, T)

        weighted_loss = torch.sum(weights * residuals**2)

        # Compute gradient ascent on attention_net
        grads = torch.autograd.grad(weighted_loss, self.attention_net.parameters(),
                                    retain_graph=True, allow_unused=True)

        with torch.no_grad():
            for param, grad in zip(self.attention_net.parameters(), grads):
                if grad is not None:
                    param = param + self.lr * grad  # ascent step

        self.iter += 1
        return weighted_loss
    




class ResidualAttentionMLP(nn.Module):
    def __init__(self, hidden_dim=16, lr=1e-3, temperature=1.0):
        """
        Residual attention using an MLP over enriched residual features.
        Args:
            hidden_dim (int): Number of hidden units in the MLP.
            lr (float): Learning rate for manual parameter updates (if used).
            temperature (float): Temperature scaling for softmax.
        """
        super().__init__()
        self.lr = lr
        self.temperature = temperature

        # MLP for attention weights
        self.attention_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive output
        )
         # Initialize optimizer for attention_net
        self.optimizer = torch.optim.SGD(self.attention_net.parameters(), lr=self.lr)

    def forward(self, residuals):
        """
        Args:
            residuals: Tensor of shape (N, G, T) or (N, G), raw residuals (not squared).
        Returns:
            weighted_loss: Scalar, sum of weighted squared residuals.
        """
        # Handle 2D residuals by adding a term dimension
        if residuals.dim() == 2:  # (N, G) -> (N, G, 1)
            residuals = residuals.unsqueeze(-1)

        # Build features: [r, r^2, log(1+|r|)]
        r = residuals
        features = torch.stack([r, r.pow(2), torch.log1p(r.abs())], dim=-1)  # (N, G, T, 3)

        # Flatten for MLP: (N*G*T, 3)
        flat_features = features.view(-1, 3)

        # Compute attention logits
        flat_logits = self.attention_net(flat_features)  # (N*G*T, 1)

        # Apply softmax with temperature over all elements (N*G*T)
        flat_weights = torch.softmax(flat_logits / self.temperature, dim=0)  # (N*G*T, 1)
        # print(f"==>> flat_weights: {flat_weights}")

        # Reshape to original residuals shape
        weights = flat_weights.view_as(residuals)  # (N, G, T)

        # Compute weighted loss
        weighted_loss = torch.sum(weights * residuals.pow(2))
        

        return weighted_loss
    
    def update_attention(self, residuals):
        """
        Recompute weighted loss and update attention MLP using gradient ascent.
        Args:
            residuals: Tensor (N, G, T) raw residuals.
        """
        self.optimizer.zero_grad()
        weighted_loss = self.forward(residuals)  # fresh forward pass
        (-weighted_loss).backward()
        self.optimizer.step()

    # def update_attention(self, weighted_loss):
    #     """
    #     Perform gradient ascent on attention_net parameters.
    #     Args:
    #         weighted_loss: Scalar tensor, the loss to maximize.
    #     """
    #     # Negate gradients for pde_model (descent)
    #     # for param in self.attention_net.parameters():
    #     #     if param.grad is not None:
    #     #         param.grad.neg_()

    #     self.optimizer.zero_grad()  # Clear previous gradients
    #     (-weighted_loss).backward(retain_graph=True)  # Compute gradients, retain graph
    #     self.optimizer.step()  # Update parameters (ascent)


    #     # # Compute gradients
    #     # grads = torch.autograd.grad(weighted_loss, self.attention_net.parameters(), create_graph=False,retain_graph=True)
    #     # # Update parameters using gradient ascent
    #     # with torch.no_grad():
    #     #     for param, grad in zip(self.attention_net.parameters(), grads):
    #     #         if grad is not None:
    #     #             param.add_(self.lr * grad)  # In-place update






class GroupSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, heads=1):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)
        self.out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, residuals):
        # residuals shape: (N, G, T)
        if residuals.dim() == 2:
            residuals = residuals.unsqueeze(-1)  # (N, G, 1)
        N, G, T = residuals.shape
        # Self-attention across groups G
        r = residuals.permute(0, 2, 1).reshape(N * T, G, 1)  # (N*T, G, 1)
        x = self.embed(r)  # (N*T, G, embed_dim)
        attn_out, _ = self.attn(x, x, x)  # (N*T, G, embed_dim)
        w = self.sigmoid(self.out(attn_out)).squeeze(-1)  # (N*T, G)
        weights = w.view(N, T, G).permute(0, 2, 1)  # (N, G, T)
        weighted_loss = torch.sum(weights * residuals.pow(2))
        return weighted_loss, weights.detach()

class TermSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, heads=1):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)
        self.out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, residuals):
        # residuals shape: (N, G, T)
        if residuals.dim() == 2:
            residuals = residuals.unsqueeze(-1)  # (N, G, 1)
        N, G, T = residuals.shape
        # Self-attention across terms T
        r = residuals.reshape(N * G, T, 1)  # (N*G, T, 1)
        x = self.embed(r)  # (N*G, T, embed_dim)
        attn_out, _ = self.attn(x, x, x)  # (N*G, T, embed_dim)
        w = self.sigmoid(self.out(attn_out)).squeeze(-1)  # (N*G, T)
        weights = w.view(N, G, T)  # (N, G, T)
        weighted_loss = torch.sum(weights * residuals.pow(2))
        return weighted_loss, weights.detach()

class DualSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, heads=1):
        super().__init__()
        # Attention over groups
        self.group_attention = GroupSelfAttention(embed_dim, heads)
        # Attention over terms
        self.term_attention = TermSelfAttention(embed_dim, heads)

    def forward(self, residuals):
        # First attention over groups
        _, group_weights = self.group_attention(residuals)
        # Weight residuals by group weights
        weighted_residuals = residuals * group_weights
        # Then attention over terms
        weighted_loss, term_weights = self.term_attention(weighted_residuals)
        # Final weights combine both attentions
        final_weights = group_weights * term_weights
        return weighted_loss, final_weights.detach()

class ResidualWeightMLP(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, residuals):
        x = residuals.unsqueeze(-1).reshape(-1, 1)
        weights = self.net(x).reshape(residuals.shape)
        weighted_loss = torch.sum(weights * residuals.pow(2))
        return weighted_loss, weights.detach()
    
class ResidualWeightMLPEnriched(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        """
        MLP over enriched residual features (input_dim > 1)
        Args:
            input_dim: number of features per residual point (e.g., residual + other features)
            hidden_dim: hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, enriched_residuals):
        """
        Args:
            enriched_residuals: (N, G, T, F) tensor of features per residual point
        Returns:
            weighted_loss: scalar
            weights: (N, G, T) weights
        """
        N, G, T, F = enriched_residuals.shape
        x = enriched_residuals.reshape(-1, F)  # Flatten
        weights = self.net(x).reshape(N, G, T)
        # Let's assume the first feature in enriched_residuals is the scalar residual
        residuals = enriched_residuals[..., 0]
        weighted_loss = torch.sum(weights * residuals.pow(2))
        return weighted_loss, weights.detach()


class myResidualAttention(nn.Module):
    def __init__(self, mode='attention_group', embed_dim=16, heads=1, hidden_dim=8, enriched_input_dim=None):
        super().__init__()
        self.mode = mode
        if mode == 'attention_group':
            self.attention_net = GroupSelfAttention(embed_dim=embed_dim, heads=heads)
        elif mode == 'attention_term':
            self.attention_net = TermSelfAttention(embed_dim=embed_dim, heads=heads)
        elif mode == 'attention_dual':
            self.attention_net = DualSelfAttention(embed_dim=embed_dim, heads=heads)
        elif mode == 'linear':
            self.attention_net = ResidualWeightMLP(hidden_dim=hidden_dim)
        elif mode == 'mlp_enriched':
            if enriched_input_dim is None:
                raise ValueError("enriched_input_dim must be specified for mlp_enriched mode")
            self.attention_net = ResidualWeightMLPEnriched(input_dim=enriched_input_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError("mode must be one of ['attention_group', 'attention_term', 'attention_dual', 'linear', 'mlp_enriched']")

    def forward(self, residuals):
        return self.attention_net(residuals)
