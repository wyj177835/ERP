import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.video import r3d_18
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Dataset
class FallFrameSequenceDataset(Dataset):
    # The function is responsible for reading sample information from CSV and assembling consecutive frame images into a tensor (C, T, H, W) format that is acceptable by the model.
    def __init__(self, csv_file, root_dir, seq_len=30, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = []

        # Construct all sample paths and their corresponding labels
        for _, row in self.annotations.iterrows():
            chute = f"chute{int(row['chute']):02d}"
            cam = f"cam_{int(row['cam'])}"
            start = int(row['start'])
            end = int(row['end'])
            label = int(row['label'])

            for i in range(start, end - seq_len + 1, seq_len):
                frame_paths = []
                for j in range(i, i + seq_len):
                    filename = f"{j:04d}.jpg"
                    path = os.path.join(root_dir, chute, cam, filename)
                    frame_paths.append(path)
                self.samples.append((frame_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        # Obtain the information of this sample survey
        frames = []

        # Read each image one by one and convert them into tensors.
        for path in frame_paths:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames, torch.tensor(label, dtype=torch.float32)

# Model
class Pure3DCNN(nn.Module):
    def __init__(self):
        super(Pure3DCNN, self).__init__()
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)

# Evaluation
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(frames)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Confusion Matrix
def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            preds = (outputs >= 0.5).int().squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# Training Curve
def plot_training_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.tight_layout()
    plt.show()

# Training
def train(model, train_loader, val_loader, device, epochs=20, lr=1e-4, patience=5):

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_acc = 0.0

    epochs_no_improve = 0

    history = {'train_acc': [], 'val_acc': [], 'loss': []}


    for epoch in range(epochs):
        model.train()

        total_loss, correct, total = 0.0, 0, 0

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for frames, labels in progress_bar:
            # Data is sent to the device (CPU/GPU)
            frames = frames.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward propagation + loss calculation
            outputs = model(frames)
            loss = criterion(outputs, labels)

            # Backpropagation + Model Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=loss.item())


        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validate the model + Adjust the learning rate
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(avg_loss)

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'pure3dcnn_dropout.pth')
            print("New best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("\u23F9\ufe0f Early stopping triggered.")
                break
    plot_training_curves(history)

if __name__ == "__main__":

    CSV_FILE = "D:/ERP/data_tuple3.csv"
    ROOT_DIR = "D:/ERP/frames"
    BATCH_SIZE = 4
    EPOCHS = 20
    SEQ_LEN = 30
    PATIENCE = 5

    # Data enhancement
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    full_dataset = FallFrameSequenceDataset(CSV_FILE, ROOT_DIR, seq_len=SEQ_LEN, transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pure3DCNN().to(device)

    train(model, train_loader, val_loader, device, epochs=EPOCHS, lr=1e-4, patience=PATIENCE)
    plot_confusion_matrix(model, val_loader, device)