import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
        del self.model.logits

        #Comment the below 2 lines if for finetuning (else Transfer learning)
        for params in self.model.parameters():
            params.requires_grad = False

        self.last_layer = nn.Linear(in_features=self.model.last_bn.num_features, out_features=128, bias=False)
        self.bn = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward_once(self, x):
        o1 = self.model(x)
        o2 = self.last_layer(o1)
        return self.bn(o2)

    def forward(self, input1, input2):
        emb1 = self.forward_once(input1)
        emb2 = self.forward_once(input2)
        return emb1, emb2


