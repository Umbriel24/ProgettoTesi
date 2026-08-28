import os
from os import mkdir

from torchvision import transforms
from pathlib import Path



# BASE_DIR Restituisce ProgettoPytorch/
if os.path.exists("/kaggle"):
    BASE_DIR = Path("/kaggle/input/datasets/umbertogargiulo/oxfordpet")
    CIFAR_PATH = "/kaggle/input/datasets/fedesoriano/cifar100" # <-- PERCORSO CORRETTO CIFAR KAGGLE

    if(os.path.exists("/kaggle/working/persistenza")):
        pass
    else:
        mkdir("/kaggle/working/persistenza")
    PERSISTANCE_PATH = Path("/kaggle/working/persistenza")
elif os.path.exists("/content"):
    # Stiamo in google colab
    BASE_DIR = Path("/content")

    # Dataset salvati nella memoria ultra-veloce di Colab
    DATASET_PATH = Path("/content/datasets/oxfordpet/images")
    ANNOTATIONS_PATH = Path("/content/datasets/oxfordpet/annotations/list.txt")
    CIFAR_PATH = "/content/datasets/cifar-100-python" 
    
    # Persistenza salvata SU DRIVE per non perdere nulla se la sessione scade
    PERSISTANCE_PATH = Path("/content/drive/MyDrive/ProgettoTesi/persistenza")
    os.makedirs(PERSISTANCE_PATH, exist_ok=True)
    
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    PERSISTANCE_PATH = BASE_DIR / "persistance"
    CIFAR_PATH = str(BASE_DIR / "cifar-100-python") # <-- AGGIUNTO PER LOCALE

# Ricerca ricorsiva (Modificata con default "None" per evitare StopIteration)
ANNOTATIONS_PATH = next(BASE_DIR.glob("**/list.txt"), None)
temp_IMAGES_PATH = next(BASE_DIR.glob("**/Abyssinian_1.jpg"), None)

if temp_IMAGES_PATH is not None:
    IMAGES_PATH = temp_IMAGES_PATH.parent
else:
    IMAGES_PATH = None
    print("ATTENZIONE: Oxford-Pets non trovato in questa directory. (Ignora se usi solo CIFAR)")

# IPER PARAMETRI
SEED = 777
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# PRE-PROCESSING E DATA TRANSFORM
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Trasformazione per il set di TRAINING
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(), # Converte l'immagine PIL in tensore
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Centra e normalizza
])

# Trasformazione set validazione e test
VAL_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# Iperparametri
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
ALPHA = 1.0 # peso loss micro
BETA = 0.5 # peso loss macro
PRETRAINED = True
