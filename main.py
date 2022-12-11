import torch
import opacus
from data import generate_cifar_datasets, generate_mnist_datasets, generate_emnist_datasets
from models import WideResnet50
from training.train import train, test, accuracy
import torchvision
import torchvision.transforms as transforms
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
# from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

# Edit here if running locally
MAC_M1 = False

########################## Data Loading #################################
# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
DATASET = "cifar10"
BATCH_SIZE = 32
MAX_PHYSICAL_BATCH_SIZE = 128

if DATASET == "cifar10": 
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010) 
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.RandomCrop(size=(1,1))
                ])
    train_loader, val_loader, test_loader = generate_cifar_datasets(BATCH_SIZE, transform)

elif DATASET == "mnist": 
    MNIST_MEAN = (0.1307,0.1307,0.1307)
    MNIST_STD_DEV = (0.3081,0.3081,0.3081)
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(3),
                    transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV),
                ])
    train_loader, val_loader, test_loader = generate_mnist_datasets(BATCH_SIZE, transform)

elif DATASET == "emnist": 
    MNIST_MEAN = (0.1307,0.1307,0.1307)
    MNIST_STD_DEV = (0.3081,0.3081,0.3081)
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(3),
                    transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV),
                ])
    train_loader, val_loader, test_loader = generate_emnist_datasets(BATCH_SIZE, transform)

else: 
    raise Exception("IllegalArgument: Invalid dataset (should be 'cifar10', 'mnist', or 'emnist')")

########################## Model Instantiation & Fixing #################################
# model = WideResnet50()
model = torchvision.models.resnet18(num_classes=10)

#hyperparameters
MAX_GRAD_NORM = 1.0 #maximum L2 norm of per-sample gradients before they are aggregated by the averaging step
EPSILON = 6.0
DELTA = 1e-5 #target of the (epsilon, delta)-DP guarantee. Generally, should be set less than inverse of the size of the training dataset. 
EPOCHS = 100
LR = 1e-1

# "Fix" model since BatchNorm is not DP-compatible
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)

# Send to GPU
if MAC_M1: 
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else: 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device = {DEVICE}")
model = model.to(DEVICE)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

########################## Privacy Engine #################################

privacy_engine = PrivacyEngine()

# Use tensorflow_privacy to compute noise_multiplier
# NOISE_LOWERBOUND = 1e-6  # outputs noise_multiplier=0 if init_epsilon < target_epsilon
# NOISE_MULTIPLIER = compute_noise(len(train_loader), 
#                                  BATCH_SIZE, 
#                                  EPSILON, 
#                                  EPOCHS, 
#                                  DELTA, 
#                                  NOISE_LOWERBOUND)

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

########################## Training #################################
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, epoch, DEVICE, privacy_engine, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA)
    test(model, test_loader, DEVICE)


