"""
Downstream task evaluation for medical image synthesis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm


class SegmentationEvaluator:
    """
    Evaluator for downstream segmentation tasks.
    """
    
    def __init__(
        self,
        device="cuda",
        model=None,
        in_channels=3,
        out_channels=1,
        learning_rate=1e-4,
        model_class="unet"
    ):
        """
        Initialize the segmentation evaluator.
        
        Args:
            device (str): Device to use
            model (nn.Module, optional): Pre-trained model
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            learning_rate (float): Learning rate
            model_class (str): Model class to use ("unet" or "other")
        """
        self.device = device
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            if model_class == "unet":
                self.model = UNet(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                )
            else:
                raise ValueError(f"Unknown model class: {model_class}")
        
        self.model = self.model.to(device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = DiceLoss(sigmoid=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=100):
        """
        Train the segmentation model.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            val_dataloader (DataLoader, optional): Validation dataloader
            num_epochs (int): Number of epochs to train for
            
        Returns:
            dict: Dictionary with training metrics
        """
        best_val_dice = 0.0
        training_metrics = {
            "train_loss": [],
            "val_dice": []
        }
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})
            
            training_metrics["train_loss"].append(epoch_loss / len(train_dataloader))
            
            # Validation
            if val_dataloader is not None:
                val_dice = self.evaluate(val_dataloader)
                training_metrics["val_dice"].append(val_dice)
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}, Val Dice: {val_dice:.4f}")
                
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    # Save best model
                    torch.save(self.model.state_dict(), "best_segmentation_model.pth")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        return training_metrics
    
    def evaluate(self, dataloader):
        """
        Evaluate the segmentation model.
        
        Args:
            dataloader (DataLoader): Dataloader with images and masks
            
        Returns:
            float: Mean Dice score
        """
        self.model.eval()
        self.metric.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                
                # Compute metric
                self.metric(y_pred=outputs, y=masks)
        
        # Aggregate the metric
        result = self.metric.aggregate().item()
        return result


class ClassificationEvaluator:
    """
    Evaluator for downstream classification tasks.
    """
    
    def __init__(
        self,
        device="cuda",
        model=None,
        num_classes=2,
        learning_rate=1e-4,
        pretrained=True
    ):
        """
        Initialize the classification evaluator.
        
        Args:
            device (str): Device to use
            model (nn.Module, optional): Pre-trained model
            num_classes (int): Number of classes
            learning_rate (float): Learning rate
            pretrained (bool): Whether to use pre-trained weights
        """
        self.device = device
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
            # Modify the final layer for our number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.model = self.model.to(device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=100):
        """
        Train the classification model.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            val_dataloader (DataLoader, optional): Validation dataloader
            num_epochs (int): Number of epochs to train for
            
        Returns:
            dict: Dictionary with training metrics
        """
        best_val_acc = 0.0
        training_metrics = {
            "train_loss": [],
            "val_acc": []
        }
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})
            
            training_metrics["train_loss"].append(epoch_loss / len(train_dataloader))
            
            # Validation
            if val_dataloader is not None:
                val_acc = self.evaluate(val_dataloader)
                training_metrics["val_acc"].append(val_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}, Val Accuracy: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model
                    torch.save(self.model.state_dict(), "best_classification_model.pth")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        return training_metrics
    
    def evaluate(self, dataloader):
        """
        Evaluate the classification model.
        
        Args:
            dataloader (DataLoader): Dataloader with images and labels
            
        Returns:
            float: Accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy


def compare_performance(real_data_loader, augmented_data_loader, 
                        test_loader, task="segmentation", num_epochs=50):
    """
    Compare performance of a model trained on real data vs augmented data.
    
    Args:
        real_data_loader (DataLoader): DataLoader with real training data
        augmented_data_loader (DataLoader): DataLoader with real + synthetic data
        test_loader (DataLoader): DataLoader with test data
        task (str): "segmentation" or "classification"
        num_epochs (int): Number of epochs to train for
        
    Returns:
        tuple: (Real data metrics, Augmented data metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize evaluator based on task
    if task == "segmentation":
        real_evaluator = SegmentationEvaluator(device=device)
        augmented_evaluator = SegmentationEvaluator(device=device)
    elif task == "classification":
        real_evaluator = ClassificationEvaluator(device=device)
        augmented_evaluator = ClassificationEvaluator(device=device)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Train and evaluate on real data
    print("Training on real data...")
    real_metrics = real_evaluator.train(real_data_loader, test_loader, num_epochs=num_epochs)
    real_test_metric = real_evaluator.evaluate(test_loader)
    
    # Train and evaluate on augmented data
    print("\nTraining on augmented data...")
    augmented_metrics = augmented_evaluator.train(augmented_data_loader, test_loader, num_epochs=num_epochs)
    augmented_test_metric = augmented_evaluator.evaluate(test_loader)
    
    # Print results
    metric_name = "Dice score" if task == "segmentation" else "Accuracy"
    print(f"\nTest {metric_name} with real data: {real_test_metric:.4f}")
    print(f"Test {metric_name} with augmented data: {augmented_test_metric:.4f}")
    print(f"Improvement: {augmented_test_metric - real_test_metric:.4f}")
    
    return real_metrics, augmented_metrics 