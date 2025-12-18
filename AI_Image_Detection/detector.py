import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# FREQUENCY ANALYSIS (FFT Features) 
class FFTAnalyzer:
    """Extract frequency domain features to detect AI artifacts"""
    
    @staticmethod
    def extract_fft_features(img_array):
        """
        Extract frequency domain features from image
        AI-generated images have characteristic spectral signatures
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        magnitude_shifted = np.fft.fftshift(magnitude)
        
        # Key indicators of AI generation
        high_freq_energy = np.mean(magnitude_shifted[128:, 128:])
        low_freq_energy = np.mean(magnitude_shifted[100:156, 100:156])
        contrast_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        return np.array([
            high_freq_energy,
            low_freq_energy,
            contrast_ratio,
            np.sum(magnitude_shifted)
        ])
    
    @staticmethod
    def visualize_fft(img_path, save_path=None):
        """Visualize FFT magnitude spectrum"""
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(fft))
        magnitude_log = np.log1p(magnitude)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_log, cmap='hot')
        plt.title('FFT Magnitude Spectrum')
        plt.colorbar()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()


# HYBRID DETECTOR MODEL
class HybridDetector(nn.Module):
    """
    Hybrid CNN + FFT detector for AI-generated images
    Combines spatial and frequency domain features
    """
    
    def __init__(self, backbone='resnet50'):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        else:
            base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            feat_dim = 1792
        
        # Feature extractor (remove classification head)
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            
        )
        
        # Frequency feature fusion
        self.freq_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
    
    def forward(self, x, fft_features=None):
        """Forward pass with optional frequency features"""
        features = self.features(x).squeeze(-1).squeeze(-1)
        
        if fft_features is not None:
            # Ensure fft_features is on correct device
            if isinstance(fft_features, np.ndarray):
                fft_vec = torch.tensor(fft_features, dtype=torch.float32, device=x.device)
            else:
                fft_vec = fft_features.to(x.device)
            
            fft_encoded = self.freq_fc(fft_vec.float())
            combined = torch.cat([features, fft_encoded], dim=-1)
            output = self.fusion(combined)
        else:
            output = self.classifier(features)
        
        return torch.clamp(output, 0, 1)


# GRAD-CAM FOR EXPLAINABILITY 
class GradCAM:
    """Gradient-based Class Activation Mapping for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, x, class_idx=1):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        output = self.model(x)
        
        # Compute gradients
        target = output[:, class_idx] if output.dim() > 1 else output
        target.sum().backward()
        
        # Compute activation weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, 
            size=x.shape[-2:], 
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()
    
    @staticmethod
    def overlay_heatmap(img_path, heatmap, save_path=None, alpha=0.4):
        """Overlay Grad-CAM heatmap on original image"""
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        
        if save_path:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Grad-CAM')
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        
        return overlay


# DATASET
class AIImageDataset(Dataset):
    """Dataset loader for AI image detection"""
    
    def __init__(self, image_paths, labels, transform=None, use_fft=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_fft = use_fft
        self.fft_analyzer = FFTAnalyzer()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Extract FFT features if enabled
        fft_features = None
        if self.use_fft:
            img_np = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
            fft_features = self.fft_analyzer.extract_fft_features(img_np)
        
        return img, self.labels[idx], fft_features if fft_features is not None else np.zeros(4)


# TRAINING PIPELINE 
class Trainer:
    """Training manager for AI image detector"""
    
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.history = {'train_loss': [], 'val_metrics': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        preds = []         
        true_labels = []
        
        for images, labels, fft_features in train_loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device).unsqueeze(1)
            
            # Move FFT features to device
            if isinstance(fft_features, np.ndarray):
                fft_features = torch.tensor(fft_features, dtype=torch.float32, device=self.device)
            else:
                fft_features = fft_features.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with FFT features
            outputs = self.model(images, fft_features)
            loss = self.criterion(outputs, labels)
            probs = torch.sigmoid(outputs) 
            # preds.extend(probs.cpu().numpy())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        avg_loss = total_loss / len(train_loader)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        preds, true_labels = [], []
        
        with torch.no_grad():
            for images, labels, fft_features in val_loader:
                images = images.to(self.device)
                
                if isinstance(fft_features, np.ndarray):
                    fft_features = torch.tensor(fft_features, dtype=torch.float32, device=self.device)
                else:
                    fft_features = fft_features.to(self.device)
                
                outputs = self.model(images, fft_features)
                probs = torch.sigmoid(outputs) 
                preds.extend(probs.cpu().numpy())
                true_labels.extend(labels.numpy())

        preds = np.array(preds).flatten()
        preds = np.nan_to_num(np.array(preds).flatten(), nan=0.5) # Replace NaN with 0.5 (neutral)
        true_labels = np.array(true_labels).flatten()
        if len(preds) == 0:
            print("Warning: Validation loader returned 0 samples!")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5}
        try:
            auc_val = roc_auc_score(true_labels, preds)
        except:
            auc_val = 0.5
        preds = np.array(preds).flatten()
        preds_binary = (preds > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(true_labels, preds_binary),
            'auc': auc_val,
            'precision': precision_score(true_labels, preds_binary, zero_division=0),
            'recall': recall_score(true_labels, preds_binary, zero_division=0),
            'f1': f1_score(true_labels, preds_binary, zero_division=0),
            'auc': roc_auc_score(true_labels, preds)
        }
        
        self.history['val_metrics'].append(metrics)
        return metrics
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
        print(f" Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f" Model loaded from {path}")


# ==================== 6. ADVERSARIAL ROBUSTNESS ====================
class RobustnessAnalyzer:
    """Test model robustness against adversarial perturbations"""
    
    @staticmethod
    def add_gaussian_noise(img, noise_level=0.1):
        """Add Gaussian noise"""
        return np.clip(img + np.random.normal(0, noise_level, img.shape), 0, 1)
    
    @staticmethod
    def add_motion_blur(img, kernel_size=5):
        """Add motion blur"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.filter2D(img, -1, kernel) / 255.0
    
    @staticmethod
    def jpeg_compression(img, quality=50):
        """Apply JPEG compression artifacts"""
        img_uint8 = (img * 255).astype(np.uint8)
        _, encoded = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) / 255.0
    
    @staticmethod
    def test_robustness(model, test_image_path, device):
        """Test model robustness across adversarial variants"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(test_image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Original prediction
        with torch.no_grad():
            original_pred = model(img_tensor).item()
        
        results = {'original': original_pred}
        
        # Test adversarial variants
        img_np = np.array(img) / 255.0
        
        variants = {
            'gaussian_noise_0.1': RobustnessAnalyzer.add_gaussian_noise(img_np, 0.1),
            'gaussian_noise_0.2': RobustnessAnalyzer.add_gaussian_noise(img_np, 0.2),
            'motion_blur_5': RobustnessAnalyzer.add_motion_blur((img_np * 255).astype(np.uint8), 5),
            'motion_blur_9': RobustnessAnalyzer.add_motion_blur((img_np * 255).astype(np.uint8), 9),
            'jpeg_quality_80': RobustnessAnalyzer.jpeg_compression(img_np, 80),
            'jpeg_quality_50': RobustnessAnalyzer.jpeg_compression(img_np, 50),
            'jpeg_quality_30': RobustnessAnalyzer.jpeg_compression(img_np, 30),
        }
        
        for name, variant in variants.items():
            variant_img = Image.fromarray((variant * 255).astype(np.uint8))
            variant_tensor = transform(variant_img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(variant_tensor).item()
            results[name] = pred
        
        return results
    
    @staticmethod
    def analyze_robustness(robustness_results, save_path=None):
        """Visualize robustness analysis"""
        plt.figure(figsize=(12, 6))
        
        perturbations = list(robustness_results.keys())
        predictions = list(robustness_results.values())
        
        colors = ['green' if p == robustness_results['original'] else 'orange' 
                  for p in predictions]
        
        plt.bar(range(len(perturbations)), predictions, color=colors, alpha=0.7)
        plt.xticks(range(len(perturbations)), perturbations, rotation=45, ha='right')
        plt.ylabel('Prediction (0=Real, 1=AI)')
        plt.title('Adversarial Robustness Analysis')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()