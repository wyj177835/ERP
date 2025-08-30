import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
from sklearn.metrics import classification_report

# Dataset
class FallFrameSequenceDataset(Dataset):
    def __init__(self, csv_file, root_dir, seq_len=30, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = []

        for _, row in self.annotations.iterrows():
            chute = f"chute{int(row['chute']):02d}"
            cam = f"cam_{int(row['cam'])}"
            start = int(row['start'])
            end = int(row['end'])
            label = int(row['label'])

            for i in range(start, end - seq_len + 1, seq_len):
                frame_paths = [
                    os.path.join(root_dir, chute, cam, f"{j:04d}.jpg")
                    for j in range(i, i + seq_len)
                ]
                self.samples.append((frame_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [self.transform(Image.open(p).convert("RGB")) for p in frame_paths]
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

# Metrics
def print_metrics(model, dataloader, device):
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

    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=["Non-Fall", "Fall"]))

# Main
if __name__ == "__main__":
    CSV_FILE = "D:/ERP/data_tuple3.csv"
    ROOT_DIR = "D:/ERP/frames"
    SEQ_LEN = 30
    BATCH_SIZE = 4
    MODEL_PATH = "pure3dcnn_dropout.pth"

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    full_dataset = FallFrameSequenceDataset(CSV_FILE, ROOT_DIR, seq_len=SEQ_LEN, transform=val_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pure3DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print_metrics(model, val_loader, device)
