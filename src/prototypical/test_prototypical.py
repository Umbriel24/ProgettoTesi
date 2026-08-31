import torch
import torch.nn as nn
from pathlib import Path

class TestPrototypical:
    def __init__(self, model_path: Path, seed: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.seed = seed

        # 1. Carica il modello intero e mettilo in modalità valutazione (disabilita Dropout)
        print(f"Caricamento di {self.model_path.name}...")
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()

        # 2. "Amputa" la testa di classificazione trasformandola in un estrattore
        self._strip_classifier()
        
        # (Qui in futuro chiameremo i metodi per estrarre i prototipi e testare)
        print("Estrattore di feature pronto!\n")

    def _strip_classifier(self):
        """
        Sostituisce il layer fully connected finale con una funzione Identità.
        L'output della rete diventerà l'embedding puro.
        """
        nome_file = self.model_path.name.lower()
        
        if "resnet" in nome_file:
            # ResNet18 e ResNet50 usano 'fc'
            self.model.fc = nn.Identity()
        elif "efficientnet" in nome_file or "densenet" in nome_file:
            # EfficientNet e DenseNet usano 'classifier'
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"Impossibile determinare l'ultimo layer per {nome_file}")