import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from grid_universe.renderer.texture import TextureRenderer
from grid_universe.levels.grid import Level
from grid_universe.levels.convert import to_state
from grid_universe.moves import default_move_fn
from grid_universe.objectives import exit_objective_fn
from grid_universe.levels.factories import *
from grid_universe.components.properties import AppearanceName

ROOT = Path(__file__).resolve().parent
ASSET_PATH = ROOT / "data" / "assets" / "imagen1"
ASSET_ROOT = ROOT / "data" / "assets"
TILE_OUTPUT_DIR = ROOT / "data" / "generated_tiles"

def spike():
    return create_hazard(appearance=AppearanceName.SPIKE, damage=1)

def lava():
    return create_hazard(appearance=AppearanceName.LAVA, damage=1)

def wall():
    return create_wall()

def boots():
    return create_speed_effect(multiplier=2, time=5)

def coin():
    return create_coin()

def exit():
    return create_exit()

def gem():
    return create_core(required=True)

def ghost():
    return create_phasing_effect(time=5)

def human():
    return create_agent()

def box():
    return create_box(pushable=True)

def enemy():
    return create_monster(damage=3, lethal=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1)

def shield():
    return create_immunity_effect(time=5)

def key():
    return create_key(key_id="default")

def door():
    return create_door(key_id="default")

create_entity_functions = [boots, coin, exit, gem, ghost, human,
         key, lava, door, box, enemy, shield,
         spike, wall]

FUNC_TO_LABEL = {
    'boots': 'boots',
    'coin': 'coin',
    'exit': 'exit',
    'gem': 'core',
    'ghost': 'ghost',
    'human': 'agent',
    'key': 'key',
    'lava': 'hazard',
    'door': 'door_locked',
    'box': 'box',
    'enemy': 'enemy',
    'shield': 'shield',
    'spike': 'hazard',
    'wall': 'wall',
}

ALL_LABELS = [
    "floor", "wall", "exit", "agent", "key", "coin", "door_locked",
    "hazard", "enemy", "core", "box", "shield", "ghost", "boots"
]

def create_image(func=None, seed=-1):
    level = Level(width=1, height=1, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=seed)
    level.add((0,0), create_floor())
    if func is not None:
        level.add((0,0), func())
    renderer = TextureRenderer(resolution=128, asset_root=str(ASSET_ROOT))
    return renderer.render(to_state(level))

def generate_training_tiles(output_dir=TILE_OUTPUT_DIR, num_variations=200):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for label in ALL_LABELS:
        (output_path / label).mkdir(exist_ok=True)

    label_counts = {label: 0 for label in ALL_LABELS}

    print("Generating tiles from 1x1 rendered levels...")

    CRITICAL_FUNCS = {
        'human',
        'exit',
        'robot',
        'wall',
        'shield',
        'ghost',
        'boots',
        'spike',
        'lava',
        'door',
    }

    for seed in range(num_variations):
        img = create_image(func=None, seed=seed)
        img.save(output_path / "floor" / f"floor_{seed}.png")
        label_counts["floor"] += 1

    for func in create_entity_functions:
        func_name = func.__name__
        label = FUNC_TO_LABEL.get(func_name, "floor")

        variations = num_variations * 3 if func_name in CRITICAL_FUNCS else num_variations * 2

        for seed in range(variations):
            try:
                img = create_image(func=func, seed=seed)
                img.save(output_path / label / f"{func_name}_{seed}.png")
                label_counts[label] += 1
            except Exception as e:
                print(f"Error with {func_name}: {e}")

    print("\nGeneration complete!")
    print("(* = oversampled critical class)")
    for label in sorted(label_counts.keys()):
        marker = "*" if label in ["agent", "enemy", "exit", "wall"] else " "
        print(f"{marker} {label:15s}: {label_counts[label]:4d}")

    return label_counts


class RenderedTileDataset(Dataset):
    def __init__(self, root_dir=TILE_OUTPUT_DIR, size=32):
        self.root_dir = Path(root_dir)
        self.size = size
        self.samples = []
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(ALL_LABELS))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        for label_dir in self.root_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            if label_name not in self.label_to_idx:
                continue
            label_idx = self.label_to_idx[label_name]
            for img_path in label_dir.glob('*.png'):
                self.samples.append((img_path, label_idx))

        print(f"Dataset: {len(self.samples)} samples, {len(self.label_to_idx)} classes")
        counts = {lbl: 0 for lbl in ALL_LABELS}
        for _, y in self.samples:
            counts[self.idx_to_label[y]] += 1
        print("Per-label counts:", counts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.size, self.size), Image.NEAREST)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr + np.random.normal(0, 0.02, arr.shape)
        arr = np.clip(arr, 0, 1)

        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr.astype(np.float32)), label



class AugmentedTileDataset(Dataset):
    def __init__(self, base_dataset, train=True):
        self.base_dataset = base_dataset
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        if self.transform is not None:
            img = transforms.ToPILImage()(x)
            img = self.transform(img)
            x = transforms.ToTensor()(img)

        return x, y


class MediumTileCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class TinyTileCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def train_model(epochs=30, batch_size=128, lr=1e-3, model_size='medium'):
    base_dataset = RenderedTileDataset()
    num_classes = len(ALL_LABELS)

    n_train = int(0.8 * len(base_dataset))
    n_val = len(base_dataset) - n_train
    train_base, val_base = torch.utils.data.random_split(base_dataset, [n_train, n_val])

    train_ds = AugmentedTileDataset(train_base, train=True)
    val_ds = AugmentedTileDataset(val_base, train=False)

    num_workers = 0 if torch.backends.mps.is_available() else 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_size == 'tiny':
        model = TinyTileCNN(num_classes=num_classes).to(device)
    elif model_size == 'medium':
        model = MediumTileCNN(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    class_weights = torch.ones(num_classes)
    label_to_idx = base_dataset.label_to_idx

    # assign critical weights
    if 'agent' in label_to_idx:
        class_weights[label_to_idx['agent']] = 2.0
    if 'enemy' in label_to_idx:
        class_weights[label_to_idx['enemy']] = 2.0
    if 'exit' in label_to_idx:
        class_weights[label_to_idx['exit']] = 2.0
    if 'wall' in label_to_idx:
        class_weights[label_to_idx['wall']] = 2.0

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_acc = 0.0
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(epoch + total / len(train_loader))

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            train_loss += loss.item() * y.size(0)

        train_acc = 100.0 * correct / total if total else 0.0
        train_loss = train_loss / total if total else 0.0

        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                val_loss += loss.item() * y.size(0)

        val_acc = 100.0 * correct / total if total else 0.0
        val_loss = val_loss / total if total else 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.1f}% train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.1f}% val_loss={val_loss:.4f} | lr={current_lr:.2e}", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
            print(" <- Best!")
        else:
            no_improve += 1
            print()

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    if best_state is not None:
        save_dict = {
            'state_dict': best_state,
            'num_classes': num_classes,
            'model_size': model_size,
            'best_acc': best_acc
        }
        torch.save(save_dict, "best_tile_model.pth")
        print("Saved best model to best_tile_model.pth")
    return base_dataset.idx_to_label, model_size

def export_model(idx_to_label, model_path="best_tile_model.pth", model_size='medium'):
    from utils import generate_torch_loader_snippet

    num_classes = len(idx_to_label)

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        model_size = checkpoint.get('model_size', model_size)
        print(f"Loaded checkpoint with validation accuracy: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    else:
        state_dict = checkpoint

    if model_size == 'tiny':
        model = TinyTileCNN(num_classes=num_classes)
        print("Exporting TinyTileCNN model")
    elif model_size == 'medium':
        model = MediumTileCNN(num_classes=num_classes)
        print("Exporting MediumTileCNN model")

    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.randn(1, 3, 32, 32)
    code = generate_torch_loader_snippet(
        model,
        example_inputs=example_input,
        prefer="script",
        compression="lzma",
        level=9,
    )
    with open(ROOT / "tile_cnn_loader.py", "w") as f:
        f.write(code)
        f.write(f"\n\nIDX_TO_LABEL = {repr(idx_to_label)}\n")
    print(f"Exported {ROOT / 'tile_cnn_loader.py'}")

def get_entity_label(level, gx, gy):
    objs = level.objects_at((gx, gy))

    for o in objs:
        if getattr(o, "agent", None) is not None:
            return "agent"

    for o in objs:
        if getattr(o, "exit", None) is not None:
            return "exit"

    for o in objs:
        if getattr(o, "locked", None) is not None:
            return "door_locked"

    for o in objs:
        if getattr(o, "key", None) is not None:
            return "key"

    for o in objs:
        app = getattr(o, "appearance", None)
        if app is None or getattr(app, "name", None) is None:
            continue

        name = app.name.value if hasattr(app.name, "value") else str(app.name)
        name = name.upper()

        if "HUMAN" in name:
            return "agent"
        if "EXIT" in name:
            return "exit"
        if "KEY" in name:
            return "key"
        if "COIN" in name or "GEM" in name:
            return "coin"
        if "CORE" in name:
            return "core"
        if "BOX" in name or "METALBOX" in name:
            return "box"
        if "LOCKED" in name:
            return "door_locked"
        if "WALL" in name:
            return "wall"
        if "SPIKE" in name or "LAVA" in name:
            return "hazard"
        if "MONSTER" in name or "ROBOT" in name or "WOLF" in name:
            return "enemy"
        if "SHIELD" in name:
            return "shield"
        if "GHOST" in name:
            return "ghost"
        if "BOOTS" in name:
            return "boots"

    for o in objs:
        if getattr(o, "damage", None) is not None or getattr(o, "lethal_damage", None) is not None:
            return "hazard"

    for o in objs:
        if getattr(o, "collectible", None) is not None:
            if getattr(o, "required", None):
                return "core"
            return "coin"

    for o in objs:
        if getattr(o, "pushable", None) is not None:
            return "box"

    for o in objs:
        if getattr(o, "blocking", None):
            return "wall"

    return "floor"



if __name__ == '__main__':
    print("Generate tiles from 1x1 levels")
    generate_training_tiles(num_variations=400)

    print("\nTrain model on rendered tiles")
    idx_to_label, model_size = train_model(epochs=25, batch_size=128, model_size='medium')

    print("\nExport model")
    export_model(idx_to_label=idx_to_label, model_size=model_size)

    print("\nDone!")
