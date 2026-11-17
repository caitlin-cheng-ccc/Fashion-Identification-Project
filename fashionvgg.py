import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def read_records(annotations):
    "Read Fashionpedia json and take the images' category and attribute ids"
    with open(annotations, "r", encoding="utf-8") as f:
        data = json.load(f)
    category_to_index = {}
    category_names = []
    for index, category in enumerate(data["categories"]):
        category_to_index[int(category["id"])] = index
        category_names.append(category["name"].replace("_", " ").lower())

    attribute_to_index = {}
    attribute_names = []
    for index, attribute in enumerate(data["attributes"]):
        attribute_to_index[int(attribute["id"])] = index
        attribute_names.append(attribute["name"].replace("_", " ").lower())

    categories_per_image = {}
    attributes_per_image = {}
    for image in data["images"]:
        image_id = int(image["id"])
        categories_per_image[image_id] = set()
        attributes_per_image[image_id] = set()
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        categories_per_image[image_id].add(int(annotation["category_id"]))
        for attribute_id in annotation.get("attribute_ids", []):
            attributes_per_image[image_id].add(int(attribute_id))

    records = []
    for image in data["images"]:
        records.append({
            "file_name": image["file_name"],
            "category_ids": categories_per_image[image["id"]],
            "attribute_ids": attributes_per_image[image["id"]],
        })

    return records, category_to_index, attribute_to_index, category_names, attribute_names

class FashionDataset(Dataset):
    def __init__(self, records: list[dict[str, set[int]]], images_dir: Path,
                 category_to_index: dict[int, int], attribute_to_index: dict[int, int],
                 transform):
        self.records = records
        self.images_dir = images_dir
        self.category_to_index = category_to_index
        self.attribute_to_index = attribute_to_index
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.images_dir / record["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        categories = torch.zeros(len(self.category_to_index))
        attributes = torch.zeros(len(self.attribute_to_index))
        for category_id in record["category_ids"]:
            categories[self.category_to_index[category_id]] = 1.0
        for attribute_id in record["attribute_ids"]:
            attributes[self.attribute_to_index[attribute_id]] = 1.0
        return image, categories, attributes
    
class VGGClassifier(nn.Module):
    CONFIGS = {
        "vgg11": (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
        "vgg16": (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                  512, 512, 512, "M", 512, 512, 512, "M"),
    }
    def __init__(self, num_categories: int, num_attributes: int, variant: str = "vgg11", dropout: float = 0.4):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for v in self.CONFIGS[variant]:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(v)
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
                in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.category_head = nn.Linear(512, num_categories)
        self.attribute_head = nn.Linear(512, num_attributes)
        self._initialize_weight()
        
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        category_logits = self.category_head(x)
        attribute_logits = self.attribute_head(x)
        return category_logits, attribute_logits
    
def make_transforms(image_size, training):
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(steps)

def pick_device(use_cpu):
    if torch.cuda.is_available() and not use_cpu:
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def compute_loss(model, images, categories, attributes, loss_function, device):
    images = images.to(device)
    categories = categories.to(device)
    attributes = attributes.to(device)
    category_logits, attribute_logits = model(images)
    loss = loss_function(category_logits, categories) + loss_function(attribute_logits, attributes)
    return loss

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.cpu)
    records, category_map, attribute_map, category_names, attribute_names = read_records(args.annotations)
    
    random.shuffle(records)
    validation_count = max(1, int(len(records) * args.validation_fraction))
    validation_records = records[:validation_count]
    training_records = records[validation_count:]
    if len(training_records) == 0:
        raise ValueError("No training records available after split.")
    train_dataset = FashionDataset(training_records, args.images, category_map, attribute_map, transform=make_transforms(args.image_size, training=True))
    val_dataset = FashionDataset(validation_records, args.images, category_map, attribute_map, transform=make_transforms(args.image_size, training=False))
    make_transforms(args.image_size, training=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = VGGClassifier(num_categories=len(category_names), num_attributes=len(attribute_names), variant=args.variant).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, categories, attributes in train_loader:
            loss = compute_loss(model, images, categories, attributes, loss_function, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {total_loss:.4f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, categories, attributes in val_loader:
                loss = compute_loss(model, images, categories, attributes, loss_function, device)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "category_names": category_names,
                "attribute_names": attribute_names,
                "variant": args.variant,
                "image_size": args.image_size
            }, args.output)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}, Best Loss: {best_loss:.4f}")
        
def print_top(probabilities, names, top_k, threshold):
    k = min(top_k, len(names))
    top = torch.topk(probabilities, k)
    for probability, index in zip(top.values, top.indices):
        if probability.item() >= threshold:
            print(f"  {names[index]}: {probability:.4f}")

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    category_names = checkpoint["category_names"]
    attribute_names = checkpoint["attribute_names"]
    model = VGGClassifier(num_categories=len(category_names), num_attributes=len(attribute_names), variant=checkpoint["variant"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    transform = make_transforms(checkpoint["image_size"], training=False)
    with Image.open(args.image) as source:
        image = transform(source.convert("RGB"))
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        category_logits, attribute_logits = model(image)
        
    category_probabilities = torch.sigmoid(category_logits)[0]
    attribute_probabilities = torch.sigmoid(attribute_logits)[0]
    print("Category Predictions:")
    print_top(category_probabilities, category_names, top_k=args.top_k, threshold=args.threshold)
    print("Attribute Predictions:")
    print_top(attribute_probabilities, attribute_names, top_k=args.top_k, threshold=args.threshold)
            
def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--images", type=Path, required=True)
    train_parser.add_argument("--annotations", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, default=Path("vgg_fashion.pt"))
    train_parser.add_argument("--variant", choices=("vgg11", "vgg16"), default="vgg11")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--learning-rate", type=float, default=0.0001)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(function=train)
    
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--cpu", action="store_true")
    predict_parser.set_defaults(function=predict)
    args = parser.parse_args()
    args.function(args)

if __name__ == "__main__":
    main()