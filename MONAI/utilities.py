# Import necessary libraries
import torch
import os
import numpy as np
from monai.losses import DiceLoss

def dice_metric(predicted, target):
    '''
    Calculate the Dice coefficient metric between predicted and target masks.

    Args:
        predicted (torch.Tensor): Predicted mask tensor.
        target (torch.Tensor): Target mask tensor.

    Returns:
        float: Dice coefficient value.
    '''
    
    # Initialize the DiceLoss for calculating the Dice coefficient
    dice_value = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    
    # Calculate the Dice coefficient
    value = 1 - dice_value(predicted, target).item()
    return value

def train(model, data_in, loss, optim, max_epochs, model_dir):
    '''
    Train a segmentation model using the provided arguments.

    Args:
        model (torch.nn.Module): The neural network model to train.
        data_in (tuple): A tuple containing train and test data loaders (train_loader, test_loader).
        loss (callable): The loss function used for training.
        optim (torch.optim.Optimizer): The optimizer used for updating model weights.
        max_epochs (int): The maximum number of training epochs.
        model_dir (str): Directory to save training-related files.

    Returns:
        None
    '''
    device = torch.device("cuda:0")
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    for epoch in range(1, max_epochs+1):
        print("-"*10)
        print(f"Epoch {epoch}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            train_step += 1
            volume = batch_data["image"]
            label = batch_data["label"]
            volume, label = (volume.to(device), label.to(device))
            optim.zero_grad()
            outputs = model(volume)
            
            # Calculate the training loss
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(f"{train_step}/{len(train_loader) // train_loader.batch_size}, Train loss: {train_loss.item():.4f}")

            # Calculate the Dice coefficient for training
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train dice: {train_metric:.4f}')

        print('-'*20)
        
        # Calculate the mean training loss for the epoch
        train_epoch_loss /= train_step
        print(f'Epoch loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        # Calculate the mean training Dice coefficient for the epoch
        epoch_metric_train /= train_step
        print(f'Epoch metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)
        model.eval()
        with torch.no_grad():
            test_epoch_loss = test_metric = epoch_metric_test = test_step = 0

            for test_data in test_loader:
                test_step += 1
                test_volume = test_data["image"].to(device)
                test_label = test_data["label"].to(device)
                test_outputs = model(test_volume)
                
                # Calculate the test loss
                test_loss = loss(test_outputs, test_label)
                test_epoch_loss += test_loss.item()
                
                # Calculate the Dice coefficient for testing
                test_metric = dice_metric(test_outputs, test_label)
                epoch_metric_test += test_metric

            # Calculate the mean test loss for the epoch    
            test_epoch_loss /= test_step
            print(f'Test loss epoch: {test_epoch_loss:.4f}')
            save_loss_test.append(test_epoch_loss)
            np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

            # Calculate the mean test Dice coefficient for the epoch
            epoch_metric_test /= test_step
            print(f'Test dice epoch: {epoch_metric_test:.4f}')
            save_metric_test.append(epoch_metric_test)
            np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

            # Save the model checkpoint if the test Dice coefficient improved
            if epoch_metric_test > best_metric:
                best_metric = epoch_metric_test
                best_metric_epoch = epoch
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
            
            print(f"Current epoch: {epoch}\nCurrent mean dice: {test_metric:.4f}\nBest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")


    print(f"Training completed.\n Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
