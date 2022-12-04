from collections import namedtuple
from fast_pytorch_kmeans import KMeans
import torch

Codebook = namedtuple('Codebook', ['centroids', 'labels'])

class KMeansQuantizer:
    def __init__(self, model : torch.nn.Module, bitwidth=4, neglect_params=[]):
        self.codebook = self.quantize(model, bitwidth, neglect_params=neglect_params)

    def update_codebook(self, fp32_tensor: torch.Tensor, codebook: Codebook):
        """
        update the centroids in the codebook using updated fp32_tensor
        :param fp32_tensor: [torch.(cuda.)Tensor] 
        :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
        """
        n_clusters = codebook.centroids.numel()
        fp32_tensor = fp32_tensor.view(-1)
        for k in range(n_clusters):
            codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
    
    def k_means_quantize(self, fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
        """
        quantize tensor using k-means clustering
        :param fp32_tensor:
        :param bitwidth: [int] quantization bit width, default=4
        :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
        :return:
            [Codebook = (centroids, labels)]
                centroids: [torch.(cuda.)FloatTensor] the cluster centroids
                labels: [torch.(cuda.)LongTensor] cluster label tensor
        """
        if codebook is None:
            # get number of clusters based on the quantization precision
            n_clusters = 1 << bitwidth
            # use k-means to get the quantization centroids
            kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
            labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
            centroids = kmeans.centroids.to(torch.float).view(-1)
            codebook = Codebook(centroids, labels)

        # decode the codebook into k-means quantized tensor for inference
        quantized_tensor = codebook.centroids[codebook.labels]
        fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
        return codebook
   
    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    self.update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = self.k_means_quantize(
                    param, codebook=self.codebook[name])

    @torch.no_grad()
    def quantize(self, model: torch.nn.Module, bitwidth=4, neglect_params=[]):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if not name in neglect_params:
                    if name in bitwidth:
                        codebook[name] = self.k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if not name in neglect_params:
                    if param.dim() > 1:
                        codebook[name] = self.k_means_quantize(param, bitwidth=bitwidth)
        return codebook