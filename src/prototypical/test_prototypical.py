from modelscreator import ModelsCreator
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

class TestPrototypical:
    def __init__(self, model_path: Path, seed: int, support_loader, val_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.seed = seed
        self.support_loader = support_loader
        self.val_loader = val_loader
        
        print(f"Caricamento di {self.model_path.name}...")
        nome_file = self.model_path.name.lower()
        
        # Parsing architettura
        if "resnet18" in nome_file: net_type = "resnet18"
        elif "resnet50" in nome_file: net_type = "resnet50"
        elif "densenet" in nome_file: net_type = "densenet"
        elif "efficientnet" in nome_file: net_type = "efficientnet"
        elif "mlp" in nome_file: net_type = "mlp"
        else: raise ValueError(f"Architettura non riconosciuta: {nome_file}")

        # Inizializza e carica
        self.model = ModelsCreator(
            backbone_name=net_type, pretrained=True, 
            num_micro_classes=37, num_macro_classes=2
        ).to(self.device)
        
        pesi = torch.load(self.model_path, map_location=self.device)
        if isinstance(pesi, dict) and 'state_dict' not in pesi: 
            self.model.load_state_dict(pesi)
        elif isinstance(pesi, dict) and 'state_dict' in pesi: 
            self.model.load_state_dict(pesi['state_dict'])
        else: 
            self.model = pesi 
            
        self.model.eval() 
        self._strip_classifier()
        print("Estrattore di feature pronto!")

        # Esecuzione Pipeline Prototipica
        prototypes = self.compute_prototypes(self.support_loader)
        self.evaluate_prototypes(prototypes, self.val_loader)

    def _strip_classifier(self):
        """Amputa le teste Fully Connected per ottenere embedding puri"""
        self.model.fc_micro = nn.Identity()
        self.model.fc_macro = nn.Identity()

    def compute_prototypes(self, support_loader):
        """Calcola il centroide (media vettoriale) per ogni classe"""
        print("Calcolo dei centroidi nello spazio latente...")
        embeddings_per_class = {}

        with torch.no_grad():
            for batch in support_loader:
                # Il tuo PetDataset restituisce un dizionario, non una tupla!
                images = batch["image"].to(self.device)
                labels = batch["micro_label"].cpu()
                
                outputs = self.model(images)
                # ModelsCreator restituisce (out_micro, out_macro)
                features = outputs[0].cpu() if isinstance(outputs, tuple) else outputs.cpu()
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label not in embeddings_per_class:
                        embeddings_per_class[label] = []
                    embeddings_per_class[label].append(features[i])

        prototypes = {}
        for label, feature_list in embeddings_per_class.items():
            stacked_features = torch.stack(feature_list)
            prototypes[label] = torch.mean(stacked_features, dim=0)
            
        print(f"Calcolati i prototipi per {len(prototypes)} classi.")
        return prototypes

    def evaluate_prototypes(self, prototypes, val_loader):
        """Testa le immagini del Query Set calcolando la distanza euclidea dai centroidi"""
        print("Valutazione del Validation Set con distanze euclidee...")
        
        # Ordiniamo i prototipi per chiavi così da avere una matrice coerente
        classi = sorted(list(prototypes.keys()))
        prototypes_tensor = torch.stack([prototypes[c] for c in classi]).to(self.device) # Shape: [Num_Classi, Dim_Embedding]
        
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["micro_label"].to(self.device)

                outputs = self.model(images)
                features = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Calcolo Distanza Euclidea: torch.cdist calcola le distanze tra due insiemi di vettori
                # features: [Batch_Size, Dim_Embedding] | prototypes_tensor: [Num_Classi, Dim_Embedding]
                # distanze: [Batch_Size, Num_Classi]
                distanze = torch.cdist(features, prototypes_tensor, p=2.0)
                
                # La predizione è la classe con la distanza MINIMA
                _, argmin = torch.min(distanze, dim=1)
                predicted_labels = torch.tensor([classi[i] for i in argmin]).to(self.device)
                
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy Prototypical Network: {accuracy:.2f}%")
        print("-" * 50)