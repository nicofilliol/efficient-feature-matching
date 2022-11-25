import torch

class FineGrainedPruner:
    def __init__(self, model: torch.nn.Module, sparsity_dict: dict):
        """Initialization of FineGrainedPruner object.

        Args:
            model (torch.Module): model to be pruned
            sparsity_dict (dict): sparsity levels for different layers
        """
        self.model = model
        self.masks = self.prune(sparsity_dict)

    @torch.no_grad()
    def apply_masks(self):
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]
    
    @torch.no_grad()
    def prune(self, sparsity_dict):
        masks = dict()
        for name, param in self.model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                masks[name] = self.fine_grained_prune(param, sparsity_dict.get(name, 0))
        return masks

    @torch.no_grad()
    def fine_grained_prune(self, tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
        sparsity = min(max(0.0, sparsity), 1.0)
        if sparsity == 1.0:
            tensor.zero_()
            return torch.zeros_like(tensor)
        elif sparsity == 0.0:
            return torch.ones_like(tensor)

        num_elements = tensor.numel()
        num_zeros = round(num_elements * sparsity)
        importance = tensor.abs()
        threshold = importance.view(-1).kthvalue(num_zeros).values
        mask = torch.gt(importance, threshold)
        tensor.mul_(mask)

        return mask