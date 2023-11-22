from torch import nn
import torch
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, disentanglement=False, loss="mse", **kwargs):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.disentanglement = disentanglement
        self.loss = loss

    def forward(self, inputs, srl=None):
        input_shape = inputs.shape
        device = inputs.device
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim) # 10*26, 768

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        if self.disentanglement:
            srl = srl.view(-1, 2)
            # srl is [(0, 2000), ..., (2000, 4000)]
            index_arr = []
            for i in range(distances.shape[0]):
                start, end = srl[i][0], srl[i][1]
                index = torch.argmin(distances[i, start:end])
                index_arr.append(index)

            encoding_indices = torch.tensor(index_arr).unsqueeze(1).to(device)
        else:
            # Encoding torch.Size([260, 1])
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Encoding
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        if self.loss == "mse":
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
        elif self.loss == "cosine":
            a = quantized.detach().view(-1, self._embedding_dim)
            b = inputs.view(-1, self._embedding_dim)
            e_latent_loss = F.cosine_embedding_loss(a, b, target=torch.ones(a.shape[0], device=device), reduction="sum")/torch.tensor(input_shape[0], device=device)

            a1 = quantized.view(-1, self._embedding_dim)
            b1 = inputs.detach().view(-1, self._embedding_dim)
            q_latent_loss = F.cosine_embedding_loss(a1, b1, target=torch.ones(a.shape[0], device=device), reduction="sum")/torch.tensor(input_shape[0], device=device)
        else:
            e_latent_loss, q_latent_loss = 0, 0
            exit("Error: wrong loss name")

        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW quantized.permute(0, 3, 1, 2).contiguous()
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5, disentanglement=False, loss="mse", **kwargs):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self.disentanglement = disentanglement
        self.loss = loss

    def forward(self, inputs, srl=None):
        input_shape = inputs.shape
        device = inputs.device
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances torch.Size([260, 10000])
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        if self.disentanglement:
            srl = srl.view(-1, 2)
            # srl is [(0, 2000), ..., (2000, 4000)]
            index_arr = []
            for i in range(distances.shape[0]):
                start, end = srl[i][0], srl[i][1]
                index = torch.argmin(distances[i, start:end])
                index_arr.append(index)

            encoding_indices = torch.tensor(index_arr).unsqueeze(1).to(device)
        else:
            # Encoding torch.Size([260, 1])
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        if self.loss == "mse":
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        elif self.loss == "cosine":
            a = quantized.detach().view(-1, self._embedding_dim)
            b = inputs.view(-1, self._embedding_dim)
            e_latent_loss = F.cosine_embedding_loss(a, b, target=torch.ones(a.shape[0], device=device), reduction="sum")/torch.tensor(input_shape[0], device=device)
        else:
            e_latent_loss = 0
            exit("Error: wrong loss name")

        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

    def get_latent(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        return distances

    def get_indices(self, inputs, srl=None):
        input_shape = inputs.shape
        device = inputs.device
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances torch.Size([260, 10000])
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        if self.disentanglement:
            srl = srl.view(-1, 2)
            # srl is [(0, 2000), ..., (2000, 4000)]
            index_arr = []
            for i in range(distances.shape[0]):
                start, end = srl[i][0], srl[i][1]
                index = torch.argmin(distances[i, start:end])
                index_arr.append(index)

            encoding_indices = torch.tensor(index_arr).unsqueeze(1).to(device)
        else:
            # Encoding torch.Size([260, 1])
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        return encoding_indices

    def get_hidden(self, inputs, srl=None):
        input_shape = inputs.shape
        device = inputs.device
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances torch.Size([260, 10000])
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        if self.disentanglement:
            srl = srl.view(-1, 2)
            # srl is [(0, 2000), ..., (2000, 4000)]
            index_arr = []
            for i in range(distances.shape[0]):
                start, end = srl[i][0], srl[i][1]
                index = torch.argmin(distances[i, start:end])
                index_arr.append(index)

            encoding_indices = torch.tensor(index_arr).unsqueeze(1).to(device)
        else:
            # Encoding torch.Size([260, 1])
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        return quantized


if __name__ == '__main__':
    pass