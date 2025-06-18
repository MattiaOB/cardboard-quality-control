"""
Few-Shot Learning Trainer for Cardboard Defect Detection
File: algorithms/few_shot_trainer.py

FEATURES:
- Vision Transformer + Prototypical Networks for few-shot learning
- Training with data augmentation and class balancing
- Real-time inference class for production deployment
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Environment optimizations
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Check transformers availability
try:
    import transformers
    from transformers import ViTModel, ViTImageProcessor
    FSL_AVAILABLE = True
except ImportError:
    FSL_AVAILABLE = False

class CartonDefectDataset(Dataset):
    """Custom dataset for cardboard defect detection"""

    def __init__(self, dataset_path: str, transform=None, max_samples_per_class=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []
        self.class_names = ['good', 'defective']

        self._load_dataset(max_samples_per_class)

    def _load_dataset(self, max_samples_per_class):
        """Load dataset from good/defective folders"""
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.dataset_path, class_name)

            if not os.path.exists(class_path):
                continue

            # Find all image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = []

            for ext in image_extensions:
                pattern = os.path.join(class_path, f"*{ext}")
                import glob
                image_files.extend(glob.glob(pattern))
                image_files.extend(glob.glob(pattern.upper()))

            # Limit samples if specified
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                image_files = image_files[:max_samples_per_class]

            # Add to dataset
            for img_path in image_files:
                self.samples.append({
                    'path': img_path,
                    'label': class_idx,
                    'class_name': class_name
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = cv2.imread(sample['path'])
            if image is None:
                raise ValueError(f"Cannot load {sample['path']}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image)

            return image, sample['label']

        except Exception:
            # Fallback: black image
            if self.transform:
                dummy_image = self.transform(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy_image, sample['label']

class ViTPrototypicalNetwork(nn.Module):
    """Vision Transformer + Prototypical Network for cardboard defect detection"""

    def __init__(self, model_name='google/vit-base-patch16-224',
                 feature_dim=128, num_classes=2):
        super().__init__()

        if not FSL_AVAILABLE:
            raise ImportError("Transformers library not available")

        self.model_name = model_name
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Vision Transformer backbone
        self.vit = ViTModel.from_pretrained(model_name, local_files_only=False)
        self.vit_feature_dim = self.vit.config.hidden_size  # 768 for ViT-Base

        # Projection head for feature extraction
        self.projection_head = nn.Sequential(
            nn.Linear(self.vit_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        )

        # Traditional classifier for supervised training
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, pixel_values):
        """Extract features for prototypical learning"""
        vit_outputs = self.vit(pixel_values=pixel_values)

        # Use [CLS] token as global representation
        cls_features = vit_outputs.last_hidden_state[:, 0]

        # Project to reduced feature space
        features = self.projection_head(cls_features)

        # L2 normalization for prototypical learning
        features = F.normalize(features, p=2, dim=1)

        return features

    def forward(self, pixel_values, mode='classification'):
        """
        Forward pass with different modes
        mode: 'classification' for supervised training
              'features' for prototypical feature extraction
        """
        features = self.extract_features(pixel_values)

        if mode == 'features':
            return features

        # Traditional classification
        logits = self.classifier(features)
        return logits

    def compute_prototypes(self, support_features, support_labels):
        """Compute prototypes for each class"""
        prototypes = []
        unique_labels = torch.unique(support_labels)

        for label in unique_labels:
            mask = (support_labels == label)
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def prototypical_loss(self, query_features, query_labels, prototypes):
        """Calculate prototypical loss"""
        # Euclidean distances to prototypes
        distances = torch.cdist(query_features, prototypes)

        # Log-softmax for classification
        log_p_y = F.log_softmax(-distances, dim=1)

        # Negative log likelihood
        loss = F.nll_loss(log_p_y, query_labels)

        return loss

class FewShotCartonTrainer:
    """Few-Shot Learning Trainer for cardboard defect detection"""

    def __init__(self, dataset_path: str = None,
                 device='auto', model_name='google/vit-base-patch16-224'):

        if dataset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            dataset_path = os.path.join(project_root, "dataset_augmented")

        self.dataset_path = dataset_path

        # Setup models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.device = self._setup_device(device)
        self.model_name = model_name
        self.model = None
        self.is_trained = False

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Configuration
        self.config = {
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': 35,
            'patience': 5,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'weight_decay': 0.001,
            'grad_clip': 1.0,
            'warmup_epochs': 2
        }

    def _setup_device(self, device):
        """Setup device with automatic fallback"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        return torch.device(device)

    def create_transforms(self):
        """Create transforms for training and validation"""
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def load_dataset(self, max_samples_per_class=None):
        """Load and split dataset"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        good_path = os.path.join(self.dataset_path, 'good')
        defective_path = os.path.join(self.dataset_path, 'defective')

        if not os.path.exists(good_path):
            raise FileNotFoundError(f"'good' folder not found: {good_path}")

        if not os.path.exists(defective_path):
            raise FileNotFoundError(f"'defective' folder not found: {defective_path}")

        # Create transforms
        train_transform, val_transform = self.create_transforms()

        # Load full dataset
        full_dataset = CartonDefectDataset(
            self.dataset_path,
            transform=None,
            max_samples_per_class=max_samples_per_class
        )

        if len(full_dataset) == 0:
            raise ValueError("Empty dataset! Check good/defective folders")

        # Split dataset using stratified sampling
        train_data, temp_data = train_test_split(
            full_dataset.samples,
            test_size=(1 - self.config['train_split']),
            stratify=[s['label'] for s in full_dataset.samples],
            random_state=42,
            shuffle=True
        )

        val_size = self.config['val_split'] / (self.config['val_split'] + self.config['test_split'])
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            stratify=[s['label'] for s in temp_data],
            random_state=42,
            shuffle=True
        )

        # Print split information
        train_labels = [s['label'] for s in train_data]
        val_labels = [s['label'] for s in val_data]
        test_labels = [s['label'] for s in test_data]

        print(f"Train: {len(train_data)} - Good: {train_labels.count(0)} Defective: {train_labels.count(1)}")
        print(f"Val: {len(val_data)} - Good: {val_labels.count(0)} Defective: {val_labels.count(1)}")
        print(f"Test: {len(test_data)} - Good: {test_labels.count(0)} Defective: {test_labels.count(1)}")

        # Adaptive batch size
        min_class_count = min(train_labels.count(0), train_labels.count(1))
        if self.config['batch_size'] > min_class_count // 4:
            self.config['batch_size'] = max(4, min_class_count // 4)

        # Create datasets with transforms
        train_dataset = self._create_subset_dataset(train_data, train_transform)
        val_dataset = self._create_subset_dataset(val_data, val_transform)
        test_dataset = self._create_subset_dataset(test_data, val_transform)

        # Create dataloaders
        num_workers = 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False
        )

        return train_loader, val_loader, test_loader

    def _create_subset_dataset(self, samples, transform):
        """Create dataset from subset of samples"""
        class SubsetDataset(Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                image = cv2.imread(sample['path'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transform:
                    image = self.transform(image)

                return image, sample['label']

        return SubsetDataset(samples, transform)

    def create_model(self):
        """Create and initialize model"""
        if not FSL_AVAILABLE:
            raise ImportError("Transformers not available. Install: pip install transformers")

        self.model = ViTPrototypicalNetwork(
            model_name=self.model_name,
            feature_dim=128,
            num_classes=2
        ).to(self.device)

        return self.model

    def train_model(self, train_loader, val_loader, save_best=True):
        """Complete model training"""
        if self.model is None:
            self.create_model()

        # Calculate class weights for imbalanced dataset
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.tolist())

        class_counts = [all_labels.count(0), all_labels.count(1)]
        total_samples = len(all_labels)

        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (2 * count)
                class_weights.append(weight)
            else:
                class_weights.append(1.0)

        good_ratio = class_counts[0] / total_samples
        if good_ratio > 0.7:
            class_weights[1] *= 2.0  # Extra weight for defective

        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Separate parameters for different learning rates
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'projection' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        # Optimizer with differentiated learning rates
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.01, 'momentum': 0.9},
            {'params': classifier_params, 'lr': self.config['learning_rate'] * 10, 'momentum': 0.9}
        ], weight_decay=0.01)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.7
        )

        # Aggressive classifier initialization to break symmetry
        def init_classifier_aggressive(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=2.0)
                if m.bias is not None:
                    if good_ratio > 0.6:
                        torch.nn.init.constant_(m.bias, 0.5)
                    else:
                        torch.nn.init.constant_(m.bias, 0.0)

        self.model.classifier.apply(init_classifier_aggressive)

        # Early stopping variables
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        warmup_epochs = self.config.get('warmup_epochs', 2)

        # Training loop
        for epoch in range(self.config['epochs']):
            start_time = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)

            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)

            # Update scheduler
            if epoch >= warmup_epochs:
                scheduler.step()

            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_best:
                    best_model_state = self.model.state_dict().copy()
                improvement = "⬆️"
            else:
                patience_counter += 1
                improvement = ""

            # Print progress
            epoch_time = time.time() - start_time
            current_lr_classifier = optimizer.param_groups[1]['lr']

            print(f"Epoch {epoch+1:2d}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} {improvement} | "
                  f"Time: {epoch_time:.1f}s")

            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1} (best val_acc: {best_val_acc:.3f})")
                break

        # Restore best model
        if save_best and best_model_state:
            self.model.load_state_dict(best_model_state)

        self.is_trained = True
        return self.training_history

    def _train_epoch(self, dataloader, optimizer, criterion):
        """Training for one epoch"""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(images, mode='classification')
            loss = criterion(logits, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )

            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, dataloader, criterion):
        """Validation for one epoch"""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images, mode='classification')
                loss = criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate_model(self, test_loader):
        """Complete evaluation on test set"""
        if not self.is_trained:
            return None

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images, mode='classification')
                probabilities = F.softmax(logits, dim=1)

                _, predicted = torch.max(logits, 1)
                confidences = torch.max(probabilities, 1)[0]

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)

        class_names = ['Good', 'Defective']
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            output_dict=True
        )

        print(f"Test Accuracy: {accuracy:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save plots
        self._plot_confusion_matrix(cm, class_names, timestamp)
        self._plot_training_curves(timestamp)

        return {
            'accuracy': accuracy,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'confidences': all_confidences,
            'timestamp': timestamp
        }

    def _plot_confusion_matrix(self, cm, class_names, timestamp):
        """Plot and save confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            confusion_matrix_path = os.path.join(self.models_dir, f'confusion_matrix_{timestamp}.png')
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception:
            pass

    def _plot_training_curves(self, timestamp):
        """Plot and save training curves"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            epochs = range(1, len(self.training_history['train_loss']) + 1)

            # Loss curves
            ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
            ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Accuracy curves
            ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
            ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            training_curves_path = os.path.join(self.models_dir, f'training_curves_{timestamp}.png')
            plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception:
            pass

    def save_model(self, filepath=None):
        """Save complete model for deployment"""
        if not self.is_trained:
            return None

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.models_dir, f"few_shot_carton_model_{timestamp}.pt")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save complete model + metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model_name,
                'feature_dim': self.model.feature_dim,
                'num_classes': self.model.num_classes
            },
            'training_config': self.config,
            'training_history': self.training_history,
            'class_names': ['good', 'defective'],
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }

        torch.save(save_dict, filepath)

        # Save pickle backup
        pkl_path = filepath.replace('.pt', '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_dict, f)

        return filepath

    def load_model(self, filepath):
        """Load pre-trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)

        # Recreate model
        model_config = checkpoint['model_config']
        self.model = ViTPrototypicalNetwork(
            model_name=model_config['model_name'],
            feature_dim=model_config['feature_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training state
        self.config = checkpoint['training_config']
        self.training_history = checkpoint['training_history']
        self.is_trained = True

        return True

class FewShotPredictor:
    """Real-time prediction class with trained few-shot model"""

    def __init__(self, model_path: str, device='auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = ['good', 'defective']
        self.is_loaded = False

        # Transform for prediction
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.load_model()

    def _setup_device(self, device):
        """Setup device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)

    def load_model(self):
        """Load model for predictions"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Recreate model
            model_config = checkpoint['model_config']
            self.model = ViTPrototypicalNetwork(
                model_name=model_config['model_name'],
                feature_dim=model_config['feature_dim'],
                num_classes=model_config['num_classes']
            ).to(self.device)

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.class_names = checkpoint.get('class_names', ['good', 'defective'])
            self.is_loaded = True

        except Exception:
            self.is_loaded = False

    def predict(self, image: np.ndarray, return_confidence=True) -> dict:
        """
        Prediction on single image

        Args:
            image: Image numpy array (BGR or RGB)
            return_confidence: Whether to return confidence score

        Returns:
            dict with prediction results
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'class': -1,
                'class_name': 'unknown',
                'confidence': 0.0
            }

        try:
            start_time = time.time()

            # Preprocessing
            processed_image = self._preprocess_image(image)

            # Prediction
            with torch.no_grad():
                self.model.eval()

                # Forward pass
                logits = self.model(processed_image, mode='classification')
                probabilities = F.softmax(logits, dim=1)

                # Extract results
                confidence, predicted_class = torch.max(probabilities, dim=1)
                predicted_class = predicted_class.item()
                confidence_score = confidence.item()

            inference_time = (time.time() - start_time) * 1000  # ms

            result = {
                'class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence_score,
                'inference_time_ms': inference_time,
                'decision': 'ACCEPT' if predicted_class == 0 else 'REJECT',
                'probabilities': {
                    'good': probabilities[0][0].item(),
                    'defective': probabilities[0][1].item()
                }
            }

            return result

        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'class': -1,
                'class_name': 'error',
                'confidence': 0.0
            }

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for prediction"""
        # Convert BGR→RGB if necessary
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Apply transform
        tensor = self.transform(image_rgb)

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def get_model_info(self) -> dict:
        """Information about loaded model"""
        if not self.is_loaded:
            return {'error': 'Model not loaded'}

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            return {
                'model_type': 'Few-Shot ViT + Prototypical',
                'model_name': checkpoint['model_config']['model_name'],
                'feature_dim': checkpoint['model_config']['feature_dim'],
                'num_classes': checkpoint['model_config']['num_classes'],
                'class_names': checkpoint.get('class_names', self.class_names),
                'training_accuracy': max(checkpoint['training_history']['val_acc']) if checkpoint['training_history']['val_acc'] else 'N/A',
                'timestamp': checkpoint.get('timestamp', 'N/A'),
                'device': str(self.device),
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': os.path.getsize(self.model_path) / (1024*1024)
            }

        except Exception as e:
            return {
                'error': f'Error retrieving info: {str(e)}',
                'model_type': 'Few-Shot ViT + Prototypical',
                'device': str(self.device)
            }

def main_training_pipeline():
    """Complete training pipeline"""
    print("Few-Shot Carton Defect Detection Training Pipeline")
    print("=" * 60)

    # Determine correct dataset path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_path = os.path.join(project_root, "dataset_augmented")

    if not os.path.exists(dataset_path):
        local_dataset = "dataset_augmented"
        if os.path.exists(local_dataset):
            dataset_path = local_dataset
        else:
            print(f"Dataset not found: {dataset_path}")
            return False

    try:
        # Initialize trainer
        trainer = FewShotCartonTrainer(
            dataset_path=dataset_path,
            device='auto'
        )

        # Load dataset
        train_loader, val_loader, test_loader = trainer.load_dataset(
            max_samples_per_class=500
        )

        # Create model
        trainer.create_model()

        # Training
        print("\nStarting training...")
        history = trainer.train_model(train_loader, val_loader)

        # Evaluation
        print("\nFinal evaluation...")
        eval_results = trainer.evaluate_model(test_loader)

        # Save model
        print("\nSaving model...")
        model_path = trainer.save_model()

        # Final summary
        print(f"\nTraining completed!")
        print(f"Final accuracy: {eval_results['accuracy']:.4f}")
        print(f"F1-Score: {eval_results['f1_score']:.4f}")
        print(f"Model saved: {model_path}")

        return True

    except Exception as e:
        print(f"Error during training: {e}")
        return False

def quick_test_model(model_path: str, test_image_path: str = None):
    """Quick model test on single image"""
    print(f"Testing model: {model_path}")

    # Initialize predictor
    predictor = FewShotPredictor(model_path)

    if not predictor.is_loaded:
        print("Unable to load model")
        return False

    # Model info
    info = predictor.get_model_info()
    print(f"Model: {info.get('model_type', 'N/A')}")
    print(f"Training accuracy: {info.get('training_accuracy', 'N/A')}")

    # Test on image
    if test_image_path and os.path.exists(test_image_path):
        print(f"Testing image: {test_image_path}")

        test_img = cv2.imread(test_image_path)
        if test_img is None:
            print("Cannot load test image")
            return False

        result = predictor.predict(test_img)

        print(f"Result: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Decision: {result['decision']}")
        print(f"Time: {result['inference_time_ms']:.1f}ms")

    else:
        # Test with dummy image
        print("Testing with dummy image...")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = predictor.predict(dummy_img)
        print(f"Dummy test completed: {result['class_name']} ({result['confidence']:.3f})")

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Few-Shot Carton Defect Detection Trainer')
    parser.add_argument('--mode', choices=['train', 'test', 'info'], default='train',
                        help='Mode: train=train model, test=test model, info=show info')
    parser.add_argument('--model_path', type=str, help='Model path for test/info')
    parser.add_argument('--test_image', type=str, help='Image for testing')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Dataset path for training')

    args = parser.parse_args()

    if args.mode == 'train':
        success = main_training_pipeline()
        exit(0 if success else 1)

    elif args.mode == 'test':
        if not args.model_path:
            print("Specify --model_path for testing")
            exit(1)

        success = quick_test_model(args.model_path, args.test_image)
        exit(0 if success else 1)

    elif args.mode == 'info':
        if not args.model_path:
            print("Specify --model_path for info")
            exit(1)

        predictor = FewShotPredictor(args.model_path)
        if predictor.is_loaded:
            info = predictor.get_model_info()
            print("\nModel Information:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        exit(0)

    else:
        success = main_training_pipeline()
        exit(0 if success else 1)