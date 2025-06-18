"""
Few-Shot Learning Trainer per Controllo Difetti Cartone - FIXED VERSION
Progetto Universitario - Algoritmo per training e salvataggio modello
File: algorithms/few_shot_trainer_fixed.py
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

# Disabilita warnings per output più pulito
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Ottimizzazioni per Windows e HuggingFace
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import sistema esistente (con fallback)
try:
    from ..utils.image_utils import ImageProcessor
    from golden_samples import CartonModelConfig
    SYSTEM_INTEGRATION = True
except ImportError:
    print("⚠️ Sistema esistente non disponibile. Modalità standalone.")
    SYSTEM_INTEGRATION = False

# Verifica dipendenze Few-Shot Learning
try:
    import transformers
    from transformers import ViTModel, ViTImageProcessor
    FSL_AVAILABLE = True
    print("✅ Transformers library disponibile")
except ImportError:
    FSL_AVAILABLE = False
    print("❌ Transformers non disponibile. Installa: pip install transformers")

class CartonDefectDataset(Dataset):
    """
    Dataset customizzato per cartoni con difetti
    Carica da dataset_augmented/good e dataset_augmented/defective
    """

    def __init__(self, dataset_path: str, transform=None, max_samples_per_class=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []
        self.class_names = ['good', 'defective']

        self._load_dataset(max_samples_per_class)

        print(f"📊 Dataset caricato: {len(self.samples)} campioni totali")
        self._print_dataset_stats()

    def _load_dataset(self, max_samples_per_class):
        """Carica dataset da cartelle good/defective"""

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.dataset_path, class_name)

            if not os.path.exists(class_path):
                print(f"⚠️ Cartella non trovata: {class_path}")
                continue

            # Trova tutti i file immagine
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = []

            for ext in image_extensions:
                pattern = os.path.join(class_path, f"*{ext}")
                import glob
                image_files.extend(glob.glob(pattern))
                image_files.extend(glob.glob(pattern.upper()))

            # Limita numero campioni se specificato
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                image_files = image_files[:max_samples_per_class]

            # Aggiungi al dataset
            for img_path in image_files:
                self.samples.append({
                    'path': img_path,
                    'label': class_idx,
                    'class_name': class_name
                })

    def _print_dataset_stats(self):
        """Stampa statistiche dataset"""
        good_count = len([s for s in self.samples if s['label'] == 0])
        defective_count = len([s for s in self.samples if s['label'] == 1])

        print(f"   🟢 Good: {good_count}")
        print(f"   🔴 Defective: {defective_count}")
        print(f"   ⚖️ Bilanciamento: {good_count/len(self.samples)*100:.1f}% good")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Carica immagine
        try:
            image = cv2.imread(sample['path'])
            if image is None:
                raise ValueError(f"Impossibile caricare {sample['path']}")

            # Converti BGR → RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Applica transforms
            if self.transform:
                image = self.transform(image)

            return image, sample['label']

        except Exception as e:
            print(f"❌ Errore caricamento {sample['path']}: {e}")
            # Fallback: immagine nera
            if self.transform:
                dummy_image = self.transform(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy_image, sample['label']

class ViTPrototypicalNetwork(nn.Module):
    """
    Vision Transformer + Prototypical Network
    Ottimizzato per difetti cartone industriali
    """

    def __init__(self, model_name='google/vit-base-patch16-224',
                 feature_dim=128, num_classes=2):
        super().__init__()

        if not FSL_AVAILABLE:
            raise ImportError("Transformers library non disponibile")

        self.model_name = model_name
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Vision Transformer backbone
        print(f"📥 Caricamento ViT: {model_name}")
        try:
            self.vit = ViTModel.from_pretrained(model_name, local_files_only=False)
            print("✅ ViT caricato con successo")
        except Exception as e:
            print(f"⚠️ Warning durante caricamento ViT: {e}")
            self.vit = ViTModel.from_pretrained(model_name)

        self.vit_feature_dim = self.vit.config.hidden_size  # 768 per ViT-Base

        # Projection head per feature extraction
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

        # Classifier tradizionale (per training supervisionato)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        print(f"✅ Modello inizializzato: {self.count_parameters():,} parametri")

    def count_parameters(self):
        """Conta parametri trainable"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, pixel_values):
        """Estrae features per prototypical learning"""
        # ViT forward pass
        vit_outputs = self.vit(pixel_values=pixel_values)

        # Usa [CLS] token come rappresentazione globale
        cls_features = vit_outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]

        # Projection a feature space ridotto
        features = self.projection_head(cls_features)

        # Normalizzazione L2 per prototypical learning
        features = F.normalize(features, p=2, dim=1)

        return features

    def forward(self, pixel_values, mode='classification'):
        """
        Forward pass con modalità diverse
        mode: 'classification' per training supervisionato
              'features' per estrazione features prototypical
        """
        features = self.extract_features(pixel_values)

        if mode == 'features':
            return features

        # Classificazione tradizionale
        logits = self.classifier(features)
        return logits

    def compute_prototypes(self, support_features, support_labels):
        """Computa prototipi per each classe"""
        prototypes = []
        unique_labels = torch.unique(support_labels)

        for label in unique_labels:
            mask = (support_labels == label)
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def prototypical_loss(self, query_features, query_labels, prototypes):
        """Calcola loss prototypical"""
        # Distanze euclidee ai prototipi
        distances = torch.cdist(query_features, prototypes)

        # Log-softmax per classificazione
        log_p_y = F.log_softmax(-distances, dim=1)

        # Negative log likelihood
        loss = F.nll_loss(log_p_y, query_labels)

        return loss

class FewShotCartonTrainer:
    """
    Trainer per Few-Shot Learning su difetti cartone
    Combina training supervisionato + prototypical learning
    """

    def __init__(self, dataset_path: str = None,
                 device='auto', model_name='google/vit-base-patch16-224'):

        # Se non specificato, cerca dataset nella radice del progetto
        if dataset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Sali di un livello da algorithms
            dataset_path = os.path.join(project_root, "dataset_augmented")

        self.dataset_path = dataset_path

        # Setup models directory path in project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)  # Sali di un livello da algorithms
        self.models_dir = os.path.join(self.project_root, "models")

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

        self.device = self._setup_device(device)
        self.model_name = model_name

        # Modello
        self.model = None
        self.is_trained = False

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Configurazione ottimizzata per debug
        self.config = {
            'learning_rate': 0.0001,
            'batch_size': 16,        # Batch size normale per velocità
            'epochs': 35,            # Meno epoche per test rapido
            'patience': 200,           # Pazienza ridotta
            'train_split': 0.7,      # Più dati training
            'val_split': 0.15,       # Validation più piccolo
            'test_split': 0.15,      # Test più piccolo
            'weight_decay': 0.001,
            'grad_clip': 1.0,
            'warmup_epochs': 2       # Warmup ridotto
        }

        print(f"🎯 FewShotCartonTrainer inizializzato")
        print(f"   Dataset: {dataset_path}")
        print(f"   Models Dir: {self.models_dir}")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")

    def _setup_device(self, device):
        """Setup device con fallback automatico"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 GPU disponibile: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("💻 Usando CPU")

        return torch.device(device)

    def create_transforms(self):
        """Crea transforms per training e validation"""

        # Transforms per training (con augmentation più aggressiva)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Resize più grande prima del crop
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Rotazione più ampia
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),  # Occasionalmente grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)  # Random erasing per robustezza
        ])

        # Transforms per validation/test (senza augmentation)
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def load_dataset(self, max_samples_per_class=None):
        """Carica e splitta il dataset con controlli di qualità"""
        print("📂 Caricamento dataset...")

        # Verifica esistenza cartelle
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path non trovato: {self.dataset_path}")

        good_path = os.path.join(self.dataset_path, 'good')
        defective_path = os.path.join(self.dataset_path, 'defective')

        if not os.path.exists(good_path):
            raise FileNotFoundError(f"Cartella 'good' non trovata: {good_path}")

        if not os.path.exists(defective_path):
            raise FileNotFoundError(f"Cartella 'defective' non trovata: {defective_path}")

        # Crea transforms
        train_transform, val_transform = self.create_transforms()

        # Carica dataset completo
        full_dataset = CartonDefectDataset(
            self.dataset_path,
            transform=None,  # Applicheremo transforms dopo lo split
            max_samples_per_class=max_samples_per_class
        )

        if len(full_dataset) == 0:
            raise ValueError("Dataset vuoto! Verifica le cartelle good/defective")

        # 🔧 CONTROLLO QUALITÀ DATASET
        labels = [s['label'] for s in full_dataset.samples]
        good_count = labels.count(0)
        defective_count = labels.count(1)

        print(f"📊 Controllo bilanciamento:")
        print(f"   Good: {good_count} ({good_count/len(labels)*100:.1f}%)")
        print(f"   Defective: {defective_count} ({defective_count/len(labels)*100:.1f}%)")

        # Verifica bilanciamento
        ratio = good_count / defective_count if defective_count > 0 else float('inf')
        if ratio > 3 or ratio < 0.33:
            print(f"⚠️ DATASET SBILANCIATO! Ratio good/defective: {ratio:.2f}")
            print("💡 Considera di bilanciare il dataset o usare class weights")

        # Split STRATEGICO per evitare data leakage
        # Usa stratify per mantenere proporzioni in ogni split
        train_data, temp_data = train_test_split(
            full_dataset.samples,
            test_size=(1 - self.config['train_split']),
            stratify=[s['label'] for s in full_dataset.samples],
            random_state=42,
            shuffle=True  # Assicura shuffling
        )

        val_size = self.config['val_split'] / (self.config['val_split'] + self.config['test_split'])
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            stratify=[s['label'] for s in temp_data],
            random_state=42,
            shuffle=True
        )

        # 🔧 VERIFICA SPLIT QUALITY
        train_labels = [s['label'] for s in train_data]
        val_labels = [s['label'] for s in val_data]
        test_labels = [s['label'] for s in test_data]

        print(f"📊 Verifica split:")
        print(f"   Train: {len(train_data)} - Good: {train_labels.count(0)} Def: {train_labels.count(1)}")
        print(f"   Val: {len(val_data)} - Good: {val_labels.count(0)} Def: {val_labels.count(1)}")
        print(f"   Test: {len(test_data)} - Good: {test_labels.count(0)} Def: {test_labels.count(1)}")

        # Crea datasets con transforms
        train_dataset = self._create_subset_dataset(train_data, train_transform)
        val_dataset = self._create_subset_dataset(val_data, val_transform)
        test_dataset = self._create_subset_dataset(test_data, val_transform)

        # 🔧 BATCH SIZE ADATTIVO
        min_class_count = min(train_labels.count(0), train_labels.count(1))
        if self.config['batch_size'] > min_class_count // 4:
            old_batch_size = self.config['batch_size']
            self.config['batch_size'] = max(4, min_class_count // 4)
            print(f"⚠️ Batch size ridotto da {old_batch_size} a {self.config['batch_size']} per evitare batch vuoti")

        # Crea dataloaders con gestione errori migliorata
        try:
            num_workers = 0  # Forza 0 per Windows per velocità debug

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,  # Disabilita per debug
                drop_last=True  # Evita batch incompleti
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

        except Exception as e:
            print(f"⚠️ Errore creazione DataLoader, uso fallback: {e}")

            # Fallback semplice
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)

        print(f"📊 DataLoaders creati:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    def _create_subset_dataset(self, samples, transform):
        """Crea dataset da subset di campioni"""
        class SubsetDataset(Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]

                # Carica immagine
                image = cv2.imread(sample['path'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Applica transform
                if self.transform:
                    image = self.transform(image)

                return image, sample['label']

        return SubsetDataset(samples, transform)

    def create_model(self):
        """Crea e inizializza il modello"""
        print("🧠 Creazione modello...")

        if not FSL_AVAILABLE:
            raise ImportError("Transformers non disponibile. Installa: pip install transformers")

        self.model = ViTPrototypicalNetwork(
            model_name=self.model_name,
            feature_dim=128,
            num_classes=2
        ).to(self.device)

        return self.model

    def train_model(self, train_loader, val_loader, save_best=True):
        """Training completo del modello con correzioni per mode collapse"""
        print("🎯 Inizio training...")

        if self.model is None:
            self.create_model()

        # 🔧 LOSS con class weights e focal loss per combattere mode collapse
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.tolist())

        class_counts = [all_labels.count(0), all_labels.count(1)]
        total_samples = len(all_labels)

        # Class weights più aggressivi
        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (2 * count)
                class_weights.append(weight)
            else:
                class_weights.append(1.0)

        print(f"📊 Class distribution: Good={class_counts[0]}, Defective={class_counts[1]}")
        print(f"📊 Class weights: {class_weights}")

        # Se sbilanciamento grave, correggi
        good_ratio = class_counts[0] / total_samples
        if good_ratio > 0.7:
            # Dataset troppo sbilanciato verso good
            class_weights[1] *= 2.0  # Peso extra per defective
            print(f"🚨 Dataset sbilanciato ({good_ratio:.1%} good) - pesi corretti: {class_weights}")

        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        # 🔧 CRITERION con class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # 🔧 OPTIMIZER con parametri più aggressivi per rompere mode collapse
        # Separa parametri backbone da classifier
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'projection' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        # Learning rates differenziati + momentum forte
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.01, 'momentum': 0.9},
            {'params': classifier_params, 'lr': self.config['learning_rate'] * 10, 'momentum': 0.9}  # LR MOLTO ALTO
        ], weight_decay=0.01)

        # 🔧 SCHEDULER più semplice e stabile
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.7
        )

        # 🔧 INIZIALIZZAZIONE AGGRESSIVA del classifier per rompere simmetria
        def init_classifier_aggressive(m):
            if isinstance(m, nn.Linear):
                # Inizializzazione con bias verso classe minoritaria
                torch.nn.init.xavier_uniform_(m.weight, gain=2.0)  # Gain alto
                if m.bias is not None:
                    # Bias che favorisce classe defective se sbilanciata
                    if good_ratio > 0.6:
                        # Bias negativo per classe 0, positivo per classe 1
                        torch.nn.init.constant_(m.bias, 0.5)
                    else:
                        torch.nn.init.constant_(m.bias, 0.0)

        # Applica init solo al classifier
        self.model.classifier.apply(init_classifier_aggressive)

        print("🔧 Classifier inizializzato aggressivamente contro mode collapse")

        # Early stopping migliorato
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None

        # Warmup learning rate
        warmup_epochs = self.config.get('warmup_epochs', 2)

        print(f"🔧 Configurazione anti-mode-collapse:")
        print(f"   Backbone LR: {self.config['learning_rate'] * 0.01}")
        print(f"   Classifier LR: {self.config['learning_rate'] * 10}")  # 10x più alto!
        print(f"   Class weights: {class_weights}")
        print(f"   Good ratio: {good_ratio:.1%}")

        # Training loop
        for epoch in range(self.config['epochs']):
            start_time = time.time()

            # Training phase
            train_loss, train_acc, grad_norm = self._train_epoch_improved(train_loader, optimizer, criterion)

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

            # Early stopping check basato su validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    best_model_state = self.model.state_dict().copy()
                improvement = "⬆️"
            else:
                patience_counter += 1
                improvement = ""

            # Print progress dettagliato
            epoch_time = time.time() - start_time
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_classifier = optimizer.param_groups[1]['lr']

            print(f"Epoch {epoch+1:2d}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} {improvement} | "
                  f"LR: {current_lr_classifier:.6f} | Grad: {grad_norm:.3f} | "
                  f"Time: {epoch_time:.1f}s")

            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"🛑 Early stopping at epoch {epoch+1} (best val_acc: {best_val_acc:.3f})")
                break

        # Restore best model
        if save_best and best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Restored best model weights (Val Acc: {best_val_acc:.3f})")

        self.is_trained = True
        print("🎉 Training completato!")

        return self.training_history

    def _train_epoch_improved(self, dataloader, optimizer, criterion):
        """Training per una epoch con monitoring gradients e debugging"""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0
        gradient_norms = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            logits = self.model(images, mode='classification')
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping per evitare esplosione
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            gradient_norms.append(grad_norm.item())

            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Debug ogni 20 batch solo per primi 2 epoch
            if len(self.training_history['train_loss']) < 2 and batch_idx % 20 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                print(f"    Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.3f}, GradNorm={grad_norm:.3f}")

                # Debug predizioni per capire cosa sta succedendo
                if batch_idx == 0:
                    print(f"    Labels: {labels.cpu().numpy()}")
                    print(f"    Predictions: {predicted.cpu().numpy()}")
                    print(f"    Logits: {logits.detach().cpu().numpy()}")

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0

        return avg_loss, accuracy, avg_grad_norm

    def _validate_epoch(self, dataloader, criterion):
        """Validation per una epoch"""
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
        """Valutazione completa su test set"""
        print("📊 Valutazione modello su test set...")

        if not self.is_trained:
            print("⚠️ Modello non ancora addestrato")
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

        # Calcola metriche
        accuracy = accuracy_score(all_labels, all_predictions)

        # Classification report
        class_names = ['Good', 'Defective']
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            output_dict=True
        )

        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        print(f"🎯 Test Precision: {report['macro avg']['precision']:.4f}")
        print(f"🎯 Test Recall: {report['macro avg']['recall']:.4f}")
        print(f"🎯 Test F1-Score: {report['macro avg']['f1-score']:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Generate timestamp for consistent naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot and save confusion matrix and training curves in models directory
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
        """Plot confusion matrix and save in models directory"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save plot in models directory
            confusion_matrix_path = os.path.join(self.models_dir, f'confusion_matrix_{timestamp}.png')
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to avoid memory issues
            print(f"💾 Confusion matrix salvata: {confusion_matrix_path}")

        except Exception as e:
            print(f"⚠️ Errore nella creazione della confusion matrix: {e}")

    def _plot_training_curves(self, timestamp):
        """Plot training curves and save in models directory"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Loss curves
            epochs = range(1, len(self.training_history['train_loss']) + 1)
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

            # Save plot in models directory
            training_curves_path = os.path.join(self.models_dir, f'training_curves_{timestamp}.png')
            plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to avoid memory issues
            print(f"💾 Training curves salvate: {training_curves_path}")

        except Exception as e:
            print(f"⚠️ Errore nella creazione dei grafici: {e}")

    def save_model(self, filepath=None):
        """Salva modello completo per deployment"""
        if not self.is_trained:
            print("⚠️ Modello non addestrato")
            return None

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.models_dir, f"few_shot_carton_model_{timestamp}.pt")

        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Salva modello completo + metadati
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
        print(f"💾 Modello salvato: {filepath}")

        # Salva anche versione .pkl per compatibilità
        pkl_path = filepath.replace('.pt', '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"💾 Backup pickle salvato: {pkl_path}")

        return filepath

    def load_model(self, filepath):
        """Carica modello pre-addestrato"""
        print(f"📂 Caricamento modello: {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modello non trovato: {filepath}")

        # Carica checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)

        # Ricrea modello
        model_config = checkpoint['model_config']
        self.model = ViTPrototypicalNetwork(
            model_name=model_config['model_name'],
            feature_dim=model_config['feature_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)

        # Carica weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training state
        self.config = checkpoint['training_config']
        self.training_history = checkpoint['training_history']
        self.is_trained = True

        print("✅ Modello caricato con successo")
        return True

class FewShotPredictor:
    """
    Classe per predizioni real-time con modello few-shot addestrato
    Da utilizzare nell'interfaccia grafica
    """

    def __init__(self, model_path: str, device='auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = ['good', 'defective']
        self.is_loaded = False

        # Transform per predizione
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Carica modello
        self.load_model()

    def _setup_device(self, device):
        """Setup device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)

    def load_model(self):
        """Carica modello per predizioni"""
        try:
            print(f"📂 Caricamento modello: {self.model_path}")

            # Carica checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Ricrea modello
            model_config = checkpoint['model_config']
            self.model = ViTPrototypicalNetwork(
                model_name=model_config['model_name'],
                feature_dim=model_config['feature_dim'],
                num_classes=model_config['num_classes']
            ).to(self.device)

            # Carica weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Metadata
            self.class_names = checkpoint.get('class_names', ['good', 'defective'])

            self.is_loaded = True
            print("✅ Modello caricato e pronto per predizioni")

        except Exception as e:
            print(f"❌ Errore caricamento modello: {e}")
            self.is_loaded = False

    def predict(self, image: np.ndarray, return_confidence=True) -> dict:
        """
        Predizione su singola immagine

        Args:
            image: Immagine numpy array (BGR o RGB)
            return_confidence: Se restituire confidence score

        Returns:
            dict con risultati predizione
        """
        if not self.is_loaded:
            return {
                'error': 'Modello non caricato',
                'class': -1,
                'class_name': 'unknown',
                'confidence': 0.0
            }

        try:
            start_time = time.time()

            # Preprocessing
            processed_image = self._preprocess_image(image)

            # Predizione
            with torch.no_grad():
                self.model.eval()

                # Forward pass
                logits = self.model(processed_image, mode='classification')
                probabilities = F.softmax(logits, dim=1)

                # Estrai risultati
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
                'error': f'Errore predizione: {str(e)}',
                'class': -1,
                'class_name': 'error',
                'confidence': 0.0
            }

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocessing immagine per predizione"""

        # Converti BGR→RGB se necessario
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assumi BGR (OpenCV default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Applica transform
        tensor = self.transform(image_rgb)

        # Aggiungi batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def get_model_info(self) -> dict:
        """Informazioni sul modello caricato"""
        if not self.is_loaded:
            return {'error': 'Modello non caricato'}

        try:
            # Carica metadata dal checkpoint
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
                'error': f'Errore recupero info: {str(e)}',
                'model_type': 'Few-Shot ViT + Prototypical',
                'device': str(self.device)
            }

def main_training_pipeline():
    """
    Pipeline completa di training
    """
    print("🚀 PIPELINE TRAINING FEW-SHOT CARTON DEFECT DETECTION")
    print("=" * 70)

    # Determina il path corretto del dataset (radice del progetto)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Sali di un livello dalla cartella algorithms
    dataset_path = os.path.join(project_root, "dataset_augmented")

    print(f"🔍 Cercando dataset in: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trovato: {dataset_path}")
        print("💡 Verifica che esista la cartella con sottocartelle 'good' e 'defective'")
        print(f"📁 Directory corrente: {current_dir}")
        print(f"📁 Project root: {project_root}")

        # Verifica se esiste nella directory corrente (fallback)
        local_dataset = "dataset_augmented"
        if os.path.exists(local_dataset):
            print(f"✅ Trovato dataset locale: {local_dataset}")
            dataset_path = local_dataset
        else:
            return False

    try:
        # Inizializza trainer
        trainer = FewShotCartonTrainer(
            dataset_path=dataset_path,
            device='auto'
        )

        # Carica dataset
        train_loader, val_loader, test_loader = trainer.load_dataset(
            max_samples_per_class=500  # Limita per velocità sviluppo
        )

        # Crea modello
        trainer.create_model()

        # Training
        print("\n🎯 Inizio training...")
        history = trainer.train_model(train_loader, val_loader)

        # Valutazione
        print("\n📊 Valutazione finale...")
        eval_results = trainer.evaluate_model(test_loader)

        # Salva modello
        print("\n💾 Salvataggio modello...")
        model_path = trainer.save_model()

        # Summary finale
        print(f"\n🎉 TRAINING COMPLETATO!")
        print(f"📊 Accuracy finale: {eval_results['accuracy']:.4f}")
        print(f"📊 F1-Score: {eval_results['f1_score']:.4f}")
        print(f"💾 Modello salvato: {model_path}")
        print(f"📈 Grafici salvati nella cartella models:")
        print(f"   - Confusion Matrix: confusion_matrix_{eval_results['timestamp']}.png")
        print(f"   - Training Curves: training_curves_{eval_results['timestamp']}.png")

        return True

    except Exception as e:
        print(f"❌ Errore durante training: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test rapido del modello
def quick_test_model(model_path: str, test_image_path: str = None):
    """Test rapido del modello su immagine singola"""
    print(f"🧪 Test rapido modello: {model_path}")

    # Inizializza predictor
    predictor = FewShotPredictor(model_path)

    if not predictor.is_loaded:
        print("❌ Impossibile caricare modello")
        return False

    # Info modello
    info = predictor.get_model_info()
    print(f"📊 Modello: {info.get('model_type', 'N/A')}")
    print(f"📊 Accuracy training: {info.get('training_accuracy', 'N/A')}")
    print(f"📊 Parametri: {info.get('parameters', 'N/A'):,}")

    # Test su immagine
    if test_image_path and os.path.exists(test_image_path):
        print(f"🖼️ Test su immagine: {test_image_path}")

        # Carica immagine
        test_img = cv2.imread(test_image_path)
        if test_img is None:
            print("❌ Impossibile caricare immagine test")
            return False

        # Predizione
        result = predictor.predict(test_img)

        print(f"🎯 Risultato: {result['class_name']}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"🎯 Decisione: {result['decision']}")
        print(f"⏱️ Tempo: {result['inference_time_ms']:.1f}ms")

    else:
        # Test con immagine dummy
        print("🖼️ Test con immagine dummy...")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = predictor.predict(dummy_img)
        print(f"🎯 Test dummy completato: {result['class_name']} ({result['confidence']:.3f})")

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Few-Shot Carton Defect Detection Trainer - FIXED')
    parser.add_argument('--mode', choices=['train', 'test', 'info'], default='train',
                        help='Modalità: train=addestra modello, test=testa modello, info=mostra info')
    parser.add_argument('--model_path', type=str, help='Path del modello per test/info')
    parser.add_argument('--test_image', type=str, help='Immagine per test')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path dataset per training (default: cerca nella radice del progetto)')

    args = parser.parse_args()

    if args.mode == 'train':
        print("🚀 Avvio training...")
        success = main_training_pipeline()
        exit(0 if success else 1)

    elif args.mode == 'test':
        if not args.model_path:
            print("❌ Specifica --model_path per il test")
            exit(1)

        success = quick_test_model(args.model_path, args.test_image)
        exit(0 if success else 1)

    elif args.mode == 'info':
        if not args.model_path:
            print("❌ Specifica --model_path per info")
            exit(1)

        predictor = FewShotPredictor(args.model_path)
        if predictor.is_loaded:
            info = predictor.get_model_info()
            print("\n📊 INFORMAZIONI MODELLO:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        exit(0)

    else:
        # Default: training
        print("🚀 Modalità default: training")
        success = main_training_pipeline()
        exit(0 if success else 1)