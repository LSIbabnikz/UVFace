



import torch
import torchvision
from torchvision.transforms import v2 as T

from backbone.model_ediffiqa import iresnet100

class MLP(torch.nn.Module):

    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=1) -> None:
        super().__init__()

        self.l1 = torch.nn.Linear(in_dim, hidden_dim)
        self.ac = torch.nn.GELU()
        self.l2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.ac(self.l1(x))
        return self.l2(x)
    

class EXModel(torch.nn.Module):

    def __init__(self, base_model : torch.nn.Module) -> None:
        super().__init__()

        self.base_model = base_model
        self.mlp = MLP()

    def forward(self, x):
        feat = self.base_model(x)
        pred = self.mlp(feat)
        return  pred

def get_ediffiqaL():

    base_model = iresnet100()
    model = EXModel(base_model)
    model.load_state_dict(torch.load("src/quality_model/ediffiqaL.pth"))
    
    trans = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    return model, trans