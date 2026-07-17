import os
from torchvision import transforms
from pathlib import Path

# BASE_DIR Restituisce ProgettoPytorch/
if os.path.exists("/kaggle"):
    BASE_DIR = Path("/kaggle/input/datasets/umbertogargiulo/oxfordpet")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATIONS_PATH = BASE_DIR / "annotations" / "list.txt"
IMAGES_PATH = BASE_DIR / "images"
PERSISTANCE_PATH = BASE_DIR / "persistance"

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
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
ALPHA = 1.0 # peso loss micro
BETA = 0.5 # peso loss macro
PRETRAINED = True
