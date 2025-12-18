
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import json
import argparse
from detector import HybridDetector, Trainer, AIImageDataset, GradCAM, RobustnessAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIG 
CONFIG = {
    'model_name': 'resnet18',
    'batch_size': 32,
    'epochs': 5,
    'learning_rate': 1e-6,
    'weight_decay': 1e-4,
    'val_split': 0.2,
    'seed': 42,
    'num_workers': 4,
}

# UTILS
def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_image_files(directory, extensions=['jpg', 'jpeg', 'png', 'gif']):
    """Get all image files from directory"""
    img_files = []
    for ext in extensions:
        img_files.extend(Path(directory).rglob(f'*.{ext}'))
    return sorted(img_files)

def create_dataloaders(real_dir, ai_dir, batch_size=3, val_split=0.2):
    """
    Create train/val dataloaders
    real_dir: directory with real images
    ai_dir: directory with AI-generated images (StyleGAN/Stable Diffusion)
    """
    print(" Loading datasets...")
    
    # Get files
    real_files = get_image_files(real_dir)
    ai_files = get_image_files(ai_dir)
    
    print(f"   Real images: {len(real_files)}")
    print(f"   AI images: {len(ai_files)}")
    
    # Create labels (0 = real, 1 = AI)
    all_files = real_files + ai_files
    all_labels = [0] * len(real_files) + [1] * len(ai_files)

    combined = list(zip(all_files, all_labels))
    np.random.shuffle(combined) # Shuffle BEFORE splitting
    all_files, all_labels = zip(*combined)
    
    # Train/val split
    split_idx = int(len(all_files) * (1 - val_split))
    if split_idx >= len(all_files):
        split_idx = len(all_files) - 1
    train_files = all_files[:split_idx]
    train_labels = all_labels[:split_idx]
    val_files = all_files[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = AIImageDataset(train_files, train_labels, train_transform, use_fft=True)
    val_dataset = AIImageDataset(val_files, val_labels, val_transform, use_fft=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print(f" Train set: {len(train_dataset)} | Val set: {len(val_dataset)}")
    
    return train_loader, val_loader

def train_model(train_loader, val_loader, epochs=5):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Training on {device}")
    
    # Initialize model
    model = HybridDetector(backbone=CONFIG['model_name']).to(device)
    trainer = Trainer(model, device, lr=CONFIG['learning_rate'])
    
    print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_f1 = 0
    patience_counter = 0
    patience = 10
    
    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Acc':<12} {'Val F1':<12} {'Val AUC':<12}")
    print("=" * 70)
    
    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        metrics = trainer.validate(val_loader)
        
        print(f"{epoch + 1:<8} {train_loss:<15.4f} {metrics['accuracy']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
        
        # Early stopping
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            trainer.save_model('models/detector_model.pth')
            print(f"             New best F1: {best_f1:.4f} (Model saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break
    
    print("=" * 70)
    print(f"\n Training complete! Best F1: {best_f1:.4f}")
    
    return trainer, model, device

def test_generalization(model, device, test_ai_dir):
    """
    Test generalization to unseen AI models
    test_ai_dir: directory with images from unseen models (DALL·E, Midjourney, etc.)
    """
    print("\n" + "=" * 70)
    print(" Testing Generalization to Unseen Models")
    print("=" * 70)
    
    test_files = get_image_files(test_ai_dir)
    
    if len(test_files) == 0:
        print("  No test images found. Skipping generalization test.")
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    from detector import FFTAnalyzer
    fft_analyzer = FFTAnalyzer()
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for img_path in test_files:
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Get FFT features
                img_np = np.array(img)
                fft_features = torch.tensor(fft_analyzer.extract_fft_features(img_np), 
                                          dtype=torch.float32, device=device)
                
                pred = model(img_tensor.to(device).to(torch.float32), 
                        fft_features.to(device).to(torch.float32))
                predictions.append(pred)
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
    
    if predictions:
        predictions = np.array(predictions)
        print(f"\n Results on {len(predictions)} unseen model images:")
        print(f"   Mean confidence: {predictions.mean():.4f}")
        print(f"   Std deviation: {predictions.std():.4f}")
        print(f"   Min: {predictions.min():.4f} | Max: {predictions.max():.4f}")
        print(f"   % classified as AI: {(predictions > 0.5).sum() / len(predictions) * 100:.1f}%")

def visualize_training_history(trainer):
    """Visualize training metrics"""
    history = trainer.history
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], linewidth=2, color='#667eea')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Metrics
    metrics = history['val_metrics']
    if metrics:
        epochs = range(1, len(metrics) + 1)
        accuracy = [m['accuracy'] for m in metrics]
        precision = [m['precision'] for m in metrics]
        recall = [m['recall'] for m in metrics]
        f1 = [m['f1'] for m in metrics]
        
        axes[0, 1].plot(epochs, accuracy, label='Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, precision, label='Precision', linewidth=2)
        axes[0, 1].plot(epochs, recall, label='Recall', linewidth=2)
        axes[0, 1].plot(epochs, f1, label='F1', linewidth=2)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        auc = [m['auc'] for m in metrics]
        axes[1, 0].plot(epochs, auc, linewidth=2, color='#764ba2')
        axes[1, 0].set_ylabel('AUC-ROC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_title('Validation AUC-ROC')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary
        final_metrics = metrics[-1]
        summary_text = f"""
        Final Validation Metrics:
        
        Accuracy: {final_metrics['accuracy']:.4f}
        Precision: {final_metrics['precision']:.4f}
        Recall: {final_metrics['recall']:.4f}
        F1-Score: {final_metrics['f1']:.4f}
        AUC-ROC: {final_metrics['auc']:.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("\n Training history saved to models/training_history.png")
    plt.close()

# MAIN 
def main():
    parser = argparse.ArgumentParser(description='Train AI Image Detector')
    parser.add_argument('--real-dir', type=str, default='D:/iamge_classi/AI_Image_Detection/train/real', help='Path to real images')
    parser.add_argument('--ai-dir', type=str, default='D:/iamge_classi/AI_Image_Detection/train/fake', help='Path to AI images')
    parser.add_argument('--test-dir', type=str, default=None, help='Directory with unseen model images for testing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(CONFIG['seed'])
    Path('models').mkdir(exist_ok=True)
    
    print("=" * 70)
    print(" AI Image Detector - Training Script")
    print("=" * 70)
    print(f"Config: {json.dumps(CONFIG, indent=2)}\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.real_dir,
        args.ai_dir,
        batch_size=args.batch_size,
        val_split=CONFIG['val_split']
    )
    
    # Train
    trainer, model, device = train_model(train_loader, val_loader, epochs=args.epochs)
    
    # Visualize
    visualize_training_history(trainer)
    
    # Test generalization (if provided)
    if args.test_dir:
        test_generalization(model, device, args.test_dir)
    
    print("\n" + "=" * 70)
    print(" Training Pipeline Complete!")
    print("=" * 70)
    print(f"\n Model saved to: models/detector_model.pth")
    print(f" Training history saved to: models/training_history.png")
    print(f"\n To deploy, run: python app.py")

if __name__ == '__main__':
    main()