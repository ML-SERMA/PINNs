import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualWeightingAttention(nn.Module):
    def __init__(self, shape, learn_rate=1e-3, init="ones", normalize="point",device=torch.device("cpu")):
        super().__init__()
        self.learn_rate = learn_rate
        self.normalize = normalize  # can be "point", "group", "term", "flat", or None
        self.device = device 
        self.weight = nn.Parameter(self._init_weights(shape, init).to(self.device), requires_grad=True)

    # def _init_weights(self, shape, strategy):
    #     if strategy == "ones":
    #         return torch.ones(shape)
    #     elif strategy == "uniform":
    #         return torch.rand(shape)
    #     elif strategy == "softmax":
    #         return F.softmax(torch.rand(shape), dim=0)
    #     else:
    #         raise ValueError(f"Unknown init strategy: {strategy}")
        
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

    # def adaptive_softmax(self):
    #     if self.normalize is None:
    #         return  # no normalization
    #     if self.normalize == "point":
    #         self.weight.data = F.softmax(self.weight.data, dim=0)  # normalize over N
    #     elif self.normalize == "group":
    #         self.weight.data = F.softmax(self.weight.data, dim=1)  # normalize over G
    #     elif self.normalize == "term":
    #         self.weight.data = F.softmax(self.weight.data, dim=2)  # normalize over T
    #     elif self.normalize == "flat":
    #         flat = self.weight.data.view(-1)
    #         soft = F.softmax(flat, dim=0)
    #         self.weight.data = soft.view_as(self.weight.data)
    #     else:
    #         raise ValueError(f"Unknown normalization mode: {self.normalize}")
        
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
