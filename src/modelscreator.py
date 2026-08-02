import torch.nn as nn
import torchvision.models as models


class ModelsCreator(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool=True, num_micro_classes: int=37, num_macro_classes: int=2):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            _num_features = self.backbone.fc.in_features
            setattr(self.backbone, 'fc', nn.Identity())

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            _num_features = self.backbone.fc.in_features
            setattr(self.backbone, 'fc', nn.Identity())

        elif backbone_name == "densenet":
            weights = models.DenseNet161_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet161(weights=weights)
            _num_features = self.backbone.classifier.in_features
            setattr(self.backbone, 'classifier', nn.Identity())

        elif backbone_name == "efficientnet":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            _num_features = self.backbone.classifier[1].in_features
            setattr(self.backbone, 'classifier', nn.Identity())


        elif backbone_name == "mlp":
            # Usiamo AdaptiveAvgPool2d per rimpicciolire il 224x224 in arrivo dal Dataloader
            # a un 64x64 al volo, in modo da avere esattamente 12.288 features (64*64*3)

            self.backbone = nn.Sequential(
                nn.AdaptiveAvgPool2d((64, 64)),
                nn.Flatten(),
                nn.Linear(12288, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.3)
            )

            # L'ultimo layer del backbone sputa 128 features, che andranno alle 2 teste finali

            _num_features = 128

        else:
            raise ValueError(f"Backbone '{backbone_name}' non supportato. Impossibile creare una rete senza un nome valido")

        # 2 Heads (Teste con nodi fissati a 37 e 2 dai parametri default)
        self.micro_head = nn.Linear(_num_features, num_micro_classes)
        self.macro_head = nn.Linear(_num_features, num_macro_classes)

    def forward(self, x):
        feature = self.backbone(x)
        return self.micro_head(feature), self.macro_head(feature)
