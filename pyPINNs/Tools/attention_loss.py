import torch
import torch.nn as nn
import torch.nn.functional as F

class IRDR_BRDRLayer(nn.Module):
    """
    IRDR-based BRDR weighting layer for residuals of shape (N,G,C)
    """

    def __init__(
        self, 
        mode="all",            
        beta_c=0.9,            # EMA for R^4
        beta_w=0.9,            # EMA for weights
        eps=1e-8,
        update_freq=5,
    ):
        super().__init__()

        self.beta_c = beta_c
        self.beta_w = beta_w
        self.eps = eps
        self.mode = mode
        self.update_freq = update_freq
        self.step_count = 0

        # self.register_buffer("weights", torch.ones(shape, device=device))
        # self.register_buffer("ema_r4", torch.zeros(shape, device=device))
        self.weights = None
        self.ema_r4 = None

    def compute_block_stats(self, R):

        if self.mode == "group_term":
            r2 = (R**2).mean(dim=0)
            r4 = (R**4).mean(dim=0)

        elif self.mode == "group":
            r2 = (R**2).mean(dim=(0,2))
            r4 = (R**4).mean(dim=(0,2))

        elif self.mode == "term":
            r2 = (R**2).mean(dim=(0,1))
            r4 = (R**4).mean(dim=(0,1))
            
        elif self.mode == "point":
            r2 = (R**2).mean(dim=(1,2))
            r4 = (R**4).mean(dim=(1,2))

        elif self.mode == "all":
            r2 = R**2
            r4 = R**4

        else:
            raise ValueError("Unknown mode")

        return r2, r4

    def forward(self, R):
        r2, r4 = self.compute_block_stats(R)
        if self.weights is None or self.weights.shape != r2.shape:
            self.weights = torch.ones_like(r2)
            self.ema_r4 = r4.detach().clone()

        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                # Update EMA of R^4
                self.ema_r4 = self.beta_c * self.ema_r4+ (1 - self.beta_c) * r4
                # Compute IRDR
                irdr = r2 / (torch.sqrt(self.ema_r4 + self.eps))
                # Normalize so mean weight = 1
                w_ref = irdr / (irdr.mean() + self.eps)
                # Smooth weight update
                self.weights = self.beta_w * self.weights+ (1 - self.beta_w) * w_ref
                # print(f"==>> self.weights: {self.weights}")
                
        self.step_count += 1

        # Weighted loss
        weighted = self.weights * r2
        return weighted.sum()




class CrossSectionResidualWeighting(nn.Module):
    def __init__(
        self,
        Sigma_r,
        D,
        normalize="no",
        eps=1e-12,
        device=torch.device("cpu")
    ):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
        self.device = device

        # Store cross sections (no grad, fixed physics)
        self.register_buffer("Sigma_r", Sigma_r.to(device))
        self.register_buffer("D", D.to(device))

        # Initialize weight once
        self.weight = self._init_weight()

    def _init_weight(self):
        """
        Build weight tensor [N, G, 1+d]
        """
        Sigma_r = self.Sigma_r
        D = self.D

        N, G = Sigma_r.shape
        d = D.shape[-1] if D.ndim == 3 else 1  # fallback safety

        # Physics-based scaling
        w_flux = 1.0 / (Sigma_r + self.eps)   # [N, G]
        w_grad = D                            # [N, G]

        weight = torch.cat(
            [
                w_flux.unsqueeze(-1),                 # [N,G,1]
                w_grad.unsqueeze(-1).repeat(1, 1, d)  # [N,G,d]
            ],
            dim=-1
        )  # [N, G, 1+d]

        # Normalization 
        if self.normalize == "point":
            weight = weight / (weight.mean(dim=0, keepdim=True) + self.eps)
        elif self.normalize == "group":
            weight = weight / (weight.mean(dim=1, keepdim=True) + self.eps)
        elif self.normalize == "term":
            weight = weight / (weight.mean(dim=2, keepdim=True) + self.eps)
        elif self.normalize == "flat":
            weight = weight / (weight.mean() + self.eps)

        return weight

    def forward(self, residual):
        """
        residual: [N, G, 1+d]
        """
        return torch.mean(torch.sum(self.weight * residual.pow(2),dim=tuple(range(1, residual.ndim))))
# ==============================================================================================================

class ResidualWeightingAttention(nn.Module):
    def __init__(self, shape, learn_rate=1e-3, init="ones", normalize="point",device=torch.device("cpu")):
        super().__init__()
        self.learn_rate = learn_rate
        self.normalize = normalize  # can be "point", "group", "term", "flat", or None
        self.device = device 
        self.weight = nn.Parameter(self._init_weights(shape, init).to(self.device), requires_grad=True)

    def _init_weights(self, shape, strategy):
        if strategy == "ones":
            return torch.ones(shape, device=self.device, dtype=torch.float32)
        elif strategy == "uniform":
            return torch.rand(shape, device=self.device, dtype=torch.float32)
        elif strategy == "softmax":
            return F.softmax(torch.rand(shape, device=self.device, dtype=torch.float32), dim=0)
        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def forward(self, residual):
        loss_f = torch.mean(torch.sum(self.weight * residual.pow(2), dim=tuple(range(1, residual.ndim))))
        return loss_f

    
    def adaptive_softmax(self):
        """Normalizes weights using softmax along the chosen dimension."""
        if self.normalize is None:
            return

        with torch.no_grad():
            if self.normalize == "point":
                self.weight.copy_(F.softmax(self.weight, dim=0))
            elif self.normalize == "group":
                self.weight.copy_(F.softmax(self.weight, dim=1))
            elif self.normalize == "term":
                self.weight.copy_(F.softmax(self.weight, dim=2))
            elif self.normalize == "flat":
                flat = self.weight.view(-1)
                soft = F.softmax(flat, dim=0)
                self.weight.copy_(soft.view_as(self.weight))
            else:
                raise ValueError(f"Unknown normalization mode: {self.normalize}")


    # def update_attention(self, residual):
    #     """
    #     One step of gradient ascent on attention weights using squared residuals.
    #     """
    #     # loss = -torch.mean((self.weight * residual.detach())**2)
    #     loss = torch.mean(torch.sum(self.weight * residual.pow(2), dim=tuple(range(1, residual.ndim))))

    #     print("Loss shape:", loss.shape)
    #     print("Loss value:", loss.item())
    #     self.zero_grad()
    #     loss.backward()

    #     with torch.no_grad():
    #         print("Attention weights before:", self.weight)
    #         print("Grad weights before:", self.weight.grad)
    #         self.weight += self.learn_rate * self.weight.grad
    #         self.weight.grad.zero_()
    #         self.adaptive_softmax()  # normalize after update
    #         print("Attention weights after:", self.weight)

    def update_attention(self, residual):
        """
        Performs one step of gradient ascent on the attention weights using the squared residual.
        """
        self.zero_grad()

        # Detach residual to avoid backprop through model
        loss = -torch.mean(torch.sum(self.weight * residual.detach().pow(2), dim=tuple(range(1, residual.ndim))))
        loss.backward()

        with torch.no_grad():
            grad = self.weight.grad
            grad = torch.clamp(grad, -10.0, 10.0)  # optional gradient clipping
            self.weight += self.learn_rate * grad
            self.weight.grad.zero_()
            self.adaptive_softmax()


    def log_weights(self, as_numpy=True, top_k=None):
        data = self.weight.detach()
        if as_numpy:
            data = data.cpu().numpy()
        if top_k is not None:
            flat = data.view(-1)
            indices = torch.topk(flat, top_k).indices
            values = flat[indices]
            print(f"Top-{top_k} weights:", values)
        else:
            print("Attention weights:", data)
