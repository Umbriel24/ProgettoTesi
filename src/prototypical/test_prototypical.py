from DatasetLibrary.dataset_dropper import DatasetDropper
from modelscreator import ModelsCreator
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
import re

class TestPrototypical:
    def __init__(self, model_path: Path, seed: int, support_loader, val_loader, train_subset_for_dropper):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.seed = seed
        self.support_loader = support_loader
        self.val_loader = val_loader
        
        
        print(f"Caricamento di {self.model_path.name}...")
        nome_file = self.model_path.name.lower()
        
        # Ricostruzioni classi droppate con il seed
        self.dropped_classes = []
        if train_subset_for_dropper is not None:
            self.dropped_classes = self._extract_dropped_classes(nome_file, train_subset_for_dropper)

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
        self.risultati = self.evaluate_prototypes(prototypes, self.val_loader)


    def _extract_dropped_classes(self, nome_file: str, train_subset) -> list:
        """Estrae i parametri dal nome del file e ricostruisce la lista esatta delle classi eliminate."""
        match = re.search(r"percentage(\d+)_(micro|macro)_(\d+)", nome_file)
        if match:
            perc = int(match.group(1)) / 100
            tipo_drop = match.group(2)
            file_seed = int(match.group(3))
            
            dropper = DatasetDropper(train_subset, seed=file_seed)
            
            # Eseguiamo il drop corretto (Micro o Macro) in base al nome del file
            if tipo_drop == "micro":
                dropper.drop_micro(target_macro='2', percentage=perc)
            elif tipo_drop == "macro":
                dropper.drop_macro(target_macro='2', percentage=perc)
                
            # FIX CRITICO: Allineamento ID
            # Il dropper ci dà le label grezze, ma il PetDataset fa (int(label) - 1)
            # Applichiamo lo stesso offset per far combaciare i controlli!
            id_rimossi_allineati = [int(x) - 1 for x in dropper.dropped_micro_ids]
            
            print(f"[{nome_file}] Riconosciute {len(id_rimossi_allineati)} classi droppate (Drop {tipo_drop}).")
            return id_rimossi_allineati
            
        return []

        
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
        """Testa le immagini sdoppiando l'accuracy tra classi NOTE (viste nel training) e IGNOTE (droppate)."""
        print("Valutazione del Validation Set con distanze euclidee...")
        
        classi = sorted(list(prototypes.keys()))
        prototypes_tensor = torch.stack([prototypes[c] for c in classi]).to(self.device) 
        
        # Contatori separati
        correct_seen, total_seen = 0, 0
        correct_unseen, total_unseen = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["micro_label"].to(self.device)

                outputs = self.model(images)
                features = outputs[0] if isinstance(outputs, tuple) else outputs
                
                distanze = torch.cdist(features, prototypes_tensor, p=2.0)
                _, argmin = torch.min(distanze, dim=1)
                predicted_labels = torch.tensor([classi[i] for i in argmin]).to(self.device)
                
                # Assegnazione di ogni singola immagine al suo contatore
                for i in range(len(labels)):
                    vera_classe = labels[i].item()
                    predizione = predicted_labels[i].item()
                    
                    if vera_classe in self.dropped_classes:
                        total_unseen += 1
                        if predizione == vera_classe: correct_unseen += 1
                    else:
                        total_seen += 1
                        if predizione == vera_classe: correct_seen += 1

        # Calcolo finale delle percentuali
        acc_seen = (100 * correct_seen / total_seen) if total_seen > 0 else 0
        acc_unseen = (100 * correct_unseen / total_unseen) if total_unseen > 0 else 0
        acc_globale = 100 * (correct_seen + correct_unseen) / (total_seen + total_unseen)

        print(f"Accuracy GLOBALE: {acc_globale:.2f}%")
        print(f"Accuracy CLASSI NOTE (Viste nel training): {acc_seen:.2f}%")
        if self.dropped_classes:
            print(f"Accuracy CLASSI IGNOTE (Zero-Shot / Few-Shot puro): {acc_unseen:.2f}%")
        print("-" * 50)
        return acc_globale, acc_seen, acc_unseen