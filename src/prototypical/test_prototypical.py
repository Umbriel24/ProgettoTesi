from modelscreator import ModelsCreator
import torch
import torch.nn as nn
from pathlib import Path

class TestPrototypical:
    def __init__(self, model_path: Path, seed: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.seed = seed
        
        print(f"Caricamento di {self.model_path.name}...")
        
        # Estrai il nome dell'architettura dal nome del file (es: "model_resnet18_...")
        nome_file = self.model_path.name.lower()
        if "resnet18" in nome_file: net_type = "resnet18"
        elif "resnet50" in nome_file: net_type = "resnet50"
        elif "densenet" in nome_file: net_type = "densenet"
        elif "efficientnet" in nome_file: net_type = "efficientnet"
        elif "mlp" in nome_file: net_type = "mlp"
        else: raise ValueError(f"Architettura non riconosciuta dal file: {nome_file}")

        # 1. Crea la struttura della rete base usando il tuo creatore di modelli
        # Imposto i parametri di classe standard che usi nel main
        self.model = ModelsCreator(
            backbone_name=net_type, 
            pretrained=True, 
            num_micro_classes=37, 
            num_macro_classes=2
        ).to(self.device)
        
        # 2. Inietta i pesi salvati (.pt)
        # Se i pesi sono stati salvati come dizionario, usa load_state_dict
        pesi = torch.load(self.model_path, map_location=self.device)
        if isinstance(pesi, dict) and 'state_dict' not in pesi: # È uno state_dict puro
            self.model.load_state_dict(pesi)
        elif isinstance(pesi, dict) and 'state_dict' in pesi: # È un checkpoint annidato
            self.model.load_state_dict(pesi['state_dict'])
        else: # È stato salvato il modello intero (improbabile dato l'errore precedente, ma safe fallback)
            self.model = pesi 
            
        # 3. Mettilo in valutazione
        self.model.eval() 
        
        # 4. "Amputa" la testa di classificazione trasformandola in un estrattore
        self._strip_classifier()
        
        print("Estrattore di feature pronto!\n")

    def _strip_classifier(self):
        """
        Sostituisce i layer di classificazione finali con una funzione Identità.
        Supporta l'architettura multi-task (Micro/Macro) di ModelsCreator.
        """
        # Dato che ModelsCreator ha due output (micro e macro),
        # dobbiamo disabilitarli entrambi per ottenere l'embedding puro.
        self.model.fc_micro = nn.Identity()
        self.model.fc_macro = nn.Identity()
        
        # Se all'interno del ModelsCreator l'ultimo layer del backbone originale 
        # fa da collo di bottiglia, potremmo dover tagliare anche quello.
        # Ma dato che le tue FC (micro/macro) prendono in input l'output del backbone,
        # azzerando le FC otterrai direttamente i tensori del backbone estrattore!

    def compute_prototypes(self, support_loader):
        """
        Calcola i centroidi per ogni classe passando il Support Set nell'estrattore.
        Restituisce un dizionario: {id_classe: vettore_centroide_medio}
        """
        print("Calcolo dei centroidi nello spazio latente...")
        embeddings_per_class = {}

        with torch.no_grad(): # Disabilita il calcolo dei gradienti (Inferenza pura)
            for images, labels in support_loader:
                images = images.to(self.device)
                
                # Passaggio in avanti. 
                # NOTA: ModelsCreator restituisce (out_micro, out_macro).
                # Avendo disabilitato le FC, entrambi contengono l'embedding del backbone. 
                # Ne prendiamo uno dei due (il primo).
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    features = outputs[0] 
                else:
                    features = outputs
                
                # Sposta su CPU per evitare di saturare la VRAM se il dataset è grande
                features = features.cpu()
                
                # Raggruppa gli embedding per classe
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label not in embeddings_per_class:
                        embeddings_per_class[label] = []
                    embeddings_per_class[label].append(features[i])

        # Calcola la media (centroide) per ogni classe
        prototypes = {}
        for label, feature_list in embeddings_per_class.items():
            # Impila i vettori in un tensore 2D e calcola la media lungo l'asse 0
            stacked_features = torch.stack(feature_list)
            prototypes[label] = torch.mean(stacked_features, dim=0)
            
        print(f"Calcolati i prototipi per {len(prototypes)} classi.")
        return prototypes