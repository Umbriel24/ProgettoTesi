import torch.nn as nn
import torchvision.models as models

class MultiTaskPetModel(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=False):
        ## Classifica contemporanea di Micro-categoria e Macro-categoria

        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            _num_features = self.backbone.fc.in_features

            setattr(self.backbone, 'fc', nn.Identity())

        else:
            raise ValueError(f"Backbone '{backbone_name}' non supportato.")


        # 2 Heads
        self.micro_head = nn.Linear(_num_features, 37)
        self.macro_head = nn.Linear(_num_features, 2)

    def forward(self, x):
        feature = self.backbone(x)
        return self.micro_head(feature), self.macro_head(feature)
