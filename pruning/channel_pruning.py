import torch
from SuperGlue.models.superpoint import SuperPoint

class ChannelPruner:
    def __init__(self, model: SuperPoint, prune_ratio, importance_function=torch.norm):
        """Initialization of ChannelPruner object for SuperPoint model.

        Args:
            model (SuperPoint): model to be pruned
            prune_ratio (float or list): pruning ratios, either float if uniform over all layers or list with pruning ratio for each
        """
        self.model = model
        self.importance_function = importance_function

        if not importance_function is None:
            self.apply_channel_sorting()
        self.prune(prune_ratio)

    @torch.no_grad()
    def apply_masks(self):
        # No need to apply masks because channels removed
        pass
    
    @torch.no_grad()
    def prune(self, prune_ratio):
        assert isinstance(prune_ratio, (float, list))
        n_conv = len(self.model.backbone)
        
        if isinstance(prune_ratio, list):
            assert len(prune_ratio) == n_conv - 1
        else: 
            prune_ratio = [prune_ratio] * (n_conv - 1)

        all_convs = self.model.backbone
        for i_ratio, p_ratio in enumerate(prune_ratio):
            prev_conv = all_convs[i_ratio]
            next_conv = all_convs[i_ratio + 1]
            original_channels = prev_conv.out_channels  # same as next_conv.in_channels
            n_keep = int(round(original_channels * (1. - p_ratio)))

            # Prune the output of the previous conv and bn layers
            prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
            prev_conv.bias.set_(prev_conv.bias.detach()[:n_keep])

            # Prune the input of the next conv
            next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
    
    @torch.no_grad()
    def get_input_channel_importance(self, weight):
        in_channels = weight.shape[1]
        importances = []
        # Compute the importance for each input channel
        for i_c in range(weight.shape[1]):
            channel_weight = weight.detach()[:, i_c]
            importance = self.importance_function(channel_weight)
            importances.append(importance.view(1))
        return torch.cat(importances)

    @torch.no_grad()
    def apply_channel_sorting(self):
        # Fetch all the conv and bn layers from the backbone
        all_convs = self.model.backbone

        for i_conv in range(len(all_convs) - 1):
            prev_conv = all_convs[i_conv]
            next_conv = all_convs[i_conv + 1]

            importance = self.get_input_channel_importance(next_conv.weight) # importance according to input channels
            sort_idx = torch.argsort(importance, descending=True) 

            # Apply to previous and next conv
            prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 0, sort_idx))
            prev_conv.bias.copy_(torch.index_select(prev_conv.bias.detach(), 0, sort_idx))
            next_conv.weight.copy_(torch.index_select(next_conv.weight.detach(), 1, sort_idx))