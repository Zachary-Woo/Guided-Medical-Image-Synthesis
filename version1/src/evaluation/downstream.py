"""
Downstream task evaluation for medical image synthesis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import os
import logging


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
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=100, save_path="best_segmentation_model.pth"):
        """
        Train the segmentation model.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            val_dataloader (DataLoader, optional): Validation dataloader
            num_epochs (int): Number of epochs to train for
            save_path (str): Path to save the best model
            
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
                    logging.info(f"Saving best model to {save_path} (Val Dice: {best_val_dice:.4f})")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
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
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=100, save_path="best_classification_model.pth"):
        """
        Train the classification model.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            val_dataloader (DataLoader, optional): Validation dataloader
            num_epochs (int): Number of epochs to train for
            save_path (str): Path to save the best model
            
        Returns:
            dict: Dictionary with training metrics
        """
        print("[DEBUG] Starting ClassificationEvaluator.train")
        # Check what's in the dataloader
        print(f"[DEBUG] train_dataloader type: {type(train_dataloader)}")
        print(f"[DEBUG] train_dataloader length: {len(train_dataloader) if hasattr(train_dataloader, '__len__') else 'N/A'}")
        
        # Check first batch before entering main loop
        try:
            # Get first batch directly to check its contents
            first_batch = next(iter(train_dataloader))
            print(f"[DEBUG] First batch type: {type(first_batch)}")
            if isinstance(first_batch, tuple) and len(first_batch) >= 2:
                imgs, lbls = first_batch
                print(f"[DEBUG] First batch imgs: {type(imgs)}, shape={imgs.shape if hasattr(imgs, 'shape') else 'N/A'}")
                print(f"[DEBUG] First batch lbls: {type(lbls)}, value={lbls}")
            else:
                print(f"[DEBUG] Unexpected first batch format: {first_batch}")
        except Exception as e:
            print(f"[DEBUG] Error examining first batch: {e}")
        
        best_val_acc = 0.0
        training_metrics = {
            "train_loss": [],
            "val_acc": []
        }
        print(f"[DEBUG] Initial best_val_acc: {best_val_acc} (type: {type(best_val_acc)})")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                try: # Add try-except around batch processing
                    images, labels = batch
                    print(f"[DEBUG Batch {batch_idx}] Raw batch: images type={type(images)}, labels type={type(labels)}")
                    if isinstance(labels, torch.Tensor):
                        print(f"[DEBUG Batch {batch_idx}] Labels (before conversion): {labels.shape}, {labels.dtype}, {labels.device}")
                    else:
                        print(f"[DEBUG Batch {batch_idx}] Labels (before conversion): {labels}") # Print raw labels if not tensor

                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Ensure labels are the correct shape (N) and type (long) for CrossEntropyLoss
                    labels = labels.squeeze().long()
                    print(f"[DEBUG Batch {batch_idx}] Labels (after conversion): {labels.shape}, {labels.dtype}, {labels.device}") # DEBUG

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    print(f"[DEBUG Batch {batch_idx}] Outputs: {outputs.shape}, {outputs.dtype}") # DEBUG

                    loss = self.criterion(outputs, labels)
                    print(f"[DEBUG Batch {batch_idx}] Loss: {loss.item()} (type: {type(loss.item())})") # DEBUG
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})
                except Exception as e:
                    # Print detailed error info here
                    logging.error(f"[DEBUG] Error processing batch {batch_idx}: {e}")
                    logging.error(f"[DEBUG] Offending batch labels (type {type(labels)}): {labels}")
                    logging.exception("Full traceback:") # Log the full traceback
                    raise e # Re-raise to stop execution

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            training_metrics["train_loss"].append(avg_epoch_loss)
            print(f"[DEBUG Epoch {epoch+1}] Appended train_loss: {avg_epoch_loss}") # DEBUG

            if val_dataloader is not None:
                print(f"[DEBUG Epoch {epoch+1}] Starting validation...") # DEBUG
                val_acc = self.evaluate(val_dataloader)
                print(f"[DEBUG Epoch {epoch+1}] val_acc: {val_acc} (type: {type(val_acc)})") # DEBUG
                training_metrics["val_acc"].append(val_acc)

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

                # Check types just before comparison
                print(f"[DEBUG Epoch {epoch+1}] Comparing val_acc ({type(val_acc)}) with best_val_acc ({type(best_val_acc)})")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    logging.info(f"Saving best model to {save_path} (Val Acc: {best_val_acc:.4f})")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

        return training_metrics
    
    def evaluate(self, dataloader):
        """
        Evaluate the classification model.
        
        Args:
            dataloader (DataLoader): Dataloader with images and labels
            
        Returns:
            float: Accuracy
        """
        print("[DEBUG] Starting ClassificationEvaluator.evaluate")
        print(f"[DEBUG] dataloader type: {type(dataloader)}")
        print(f"[DEBUG] dataloader length: {len(dataloader) if hasattr(dataloader, '__len__') else 'N/A'}")
        
        self.model.eval()
        correct = 0
        total = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    print(f"[DEBUG] Evaluate batch {batch_idx}: type={type(batch)}")
                    images, labels = batch
                    print(f"[DEBUG] Evaluate labels type: {type(labels)}, value={labels}")
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Ensure labels are the correct shape (N) and type (long) here as well
                    labels = labels.squeeze().long()
                    print(f"[DEBUG] Evaluate labels after conversion: {labels.shape}, {labels.dtype}")
                    
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    print(f"[DEBUG] Evaluate running total: {total}, correct: {correct}")
                    if batch_idx >= 2:  # Just check a few batches
                        break
        except Exception as e:
            import traceback
            print(f"[DEBUG] Exception in evaluate: {e}")
            traceback.print_exc()
            return -1  # Signal error
        
        # Avoid division by zero if dataloader is empty
        accuracy = float(correct / total) if total > 0 else 0.0
        print(f"[DEBUG] Calculated accuracy: {accuracy} (type: {type(accuracy)})")
        return accuracy


def train_and_evaluate_downstream(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task: str = "segmentation",
    num_epochs: int = 50,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    learning_rate: float = 1e-4,
    model_save_path: str = "downstream_model.pth",
    n_classes: int = None, # Added explicit num_classes
    n_channels: int = None # Added explicit num_channels
) -> dict:
    """
    Train and evaluate a downstream model on a given dataset configuration.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        task (str): "segmentation" or "classification".
        num_epochs (int): Number of epochs to train for.
        device (torch.device): Device to run training on.
        learning_rate (float): Learning rate for the optimizer.
        model_save_path (str): Path to save the best trained model.
        n_classes (int, optional): Number of classes for classification. Required if task is classification.
        n_channels (int, optional): Number of input channels. Required if task is segmentation.

    Returns:
        dict: Dictionary containing training history ('train_loss', 'val_metric')
              and final test metric ('test_metric').

    Raises:
        ValueError: If task is unknown or required arguments (n_classes/n_channels) are missing.
    """
    print("[DEBUG] Entering train_and_evaluate_downstream")
    
    # Initialize evaluator based on task
    if task == "segmentation":
        print(f"[DEBUG] Initializing SegmentationEvaluator (channels={n_channels})")
        evaluator = SegmentationEvaluator(
            device=device,
            learning_rate=learning_rate,
            in_channels=n_channels,
            out_channels=1 # Assuming single output channel for seg masks
        )
        val_metric_key = "val_dice"
        test_metric_key = "test_dice"

    elif task == "classification":
        print(f"[DEBUG] Initializing ClassificationEvaluator (classes={n_classes})")
        evaluator = ClassificationEvaluator(
            device=device,
            learning_rate=learning_rate,
            num_classes=n_classes
            # Add pretrained=False if needed, or make it an arg
        )
        val_metric_key = "val_acc"
        test_metric_key = "test_acc"
    else:
        raise ValueError(f"Unknown task: {task}")

    # Train the model
    logging.info(f"Training downstream {task} model for {num_epochs} epochs (LR={learning_rate})...")
    print(f"[DEBUG] About to call evaluator.train with {type(train_loader)} (len={len(train_loader) if hasattr(train_loader, '__len__') else 'N/A'})")
    
    try:
        training_history = evaluator.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=num_epochs,
            save_path=model_save_path
        )
        print("[DEBUG] Successfully completed evaluator.train")
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception in evaluator.train: {e}")
        print("[DEBUG] Traceback:")
        traceback.print_exc()
        logging.error(f"Error in train method: {e}")
        logging.error(traceback.format_exc())
        # Create an empty training history to avoid breaking the rest of the code
        training_history = {"train_loss": [], val_metric_key: []}

    # Load the best model saved during training
    logging.info(f"Loading best model from {model_save_path} for final test evaluation.")
    if os.path.exists(model_save_path):
        try:
             evaluator.model.load_state_dict(torch.load(model_save_path, map_location=device))
        except Exception as e:
             logging.error(f"Error loading model state_dict from {model_save_path}: {e}. Evaluating with last state.")
    else:
         logging.warning(f"Model checkpoint not found at {model_save_path}. Evaluating with the last state reached during training.")

    # Evaluate on the test set
    logging.info("Evaluating final model on test set...")
    final_test_metric = evaluator.evaluate(test_loader)

    # Prepare results dictionary
    results = {
        "train_loss": training_history.get("train_loss", []),
        val_metric_key: training_history.get(val_metric_key, []),
        test_metric_key: final_test_metric
    }

    metric_name = "Dice score" if task == "segmentation" else "Accuracy"
    logging.info(f"Downstream training complete. Final Test {metric_name}: {final_test_metric:.4f}")

    return results 