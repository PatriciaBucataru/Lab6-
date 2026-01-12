import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.io
import numpy as np
from sklearn.metrics import roc_auc_score


data = scipy.io.loadmat('ACM.mat')
attributes = torch.FloatTensor(data['Attributes'].toarray())
adjacency = data['Network']
labels = data['Label'].flatten()


edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)

print(f"Nodes: {attributes.shape[0]}, Features: {attributes.shape[1]}, Edges: {edge_index.shape[1]}")


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, latent_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class AttributeDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=128, out_channels=None):
        super(AttributeDecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, z, edge_index):
        x = self.conv1(z, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class StructureDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(StructureDecoder, self).__init__()
        self.conv = GCNConv(latent_dim, latent_dim)
    
    def forward(self, z, edge_index):
        z = self.conv(z, edge_index)
        z = F.relu(z)
       
        adj_recon = torch.matmul(z, z.t())
        return adj_recon


class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, latent_dim=64):
        super(GraphAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.attr_decoder = AttributeDecoder(latent_dim, hidden_channels, in_channels)
        self.struct_decoder = StructureDecoder(latent_dim)
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_recon = self.attr_decoder(z, edge_index)
        adj_recon = self.struct_decoder(z, edge_index)
        return x_recon, adj_recon


def custom_loss(X, X_hat, A, A_hat, alpha=0.8):
    attr_loss = torch.norm(X - X_hat, p='fro') ** 2
    struct_loss = torch.norm(A - A_hat, p='fro') ** 2
    return alpha * attr_loss + (1 - alpha) * struct_loss


model = GraphAutoencoder(in_channels=attributes.shape[1], hidden_channels=128, latent_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

A_dense = torch.FloatTensor(adjacency.toarray())


num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    X_recon, A_recon = model(attributes, edge_index)
    

    loss = custom_loss(attributes, X_recon, A_dense, A_recon, alpha=0.8)
    
   
    loss.backward()
    optimizer.step()
    
 
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            X_recon, A_recon = model(attributes, edge_index)
            
           
            attr_errors = torch.norm(attributes - X_recon, dim=1).numpy()
            
         
            roc_auc = roc_auc_score(labels, attr_errors)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, ROC AUC: {roc_auc:.4f}")

print("Training complete!")


model.eval()
with torch.no_grad():
    X_recon, A_recon = model(attributes, edge_index)
    anomaly_scores = torch.norm(attributes - X_recon, dim=1).numpy()


top_10_indices = np.argsort(anomaly_scores)[-10:]
print(f"top 10 anomalous nodes: {top_10_indices}")