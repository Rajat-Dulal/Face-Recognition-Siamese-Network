import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Contrastive Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute the Euclidean distance between output1 and output2
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        
        # Loss calculation
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +  # Positive pair loss
            (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # Negative pair loss
        )
        return loss_contrastive