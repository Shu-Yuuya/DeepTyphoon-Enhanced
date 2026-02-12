
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import csv
from define_net_gpu import Net
from my_transform import train_transform, test_val_transform
from my_image_folder import ImageFolder

def log_to_csv(filepath, epoch, train_loss, val_mae, val_rmse, val_r2, lr):
    file_exists = os.path.isfile(filepath)
    mode = 'w' if epoch == 1 else 'a'
    write_header = (mode == 'w') or (not file_exists)
    with open(filepath, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Epoch', 'Train_Loss', 'Val_MAE', 'Val_RMSE', 'Val_R2', 'Learning_Rate'])
        writer.writerow([epoch, train_loss, val_mae, val_rmse, val_r2, lr])

def validate(dataset, network, device):
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    network.eval() 

    all_labels = []
    all_preds = []
    total_l1_loss = 0.0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = network(inputs)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            
            loss = torch.abs(labels.view(-1) - outputs.view(-1))
            total_l1_loss += loss.sum().item()

    all_labels = np.concatenate(all_labels).ravel()
    all_preds = np.concatenate(all_preds).ravel()

    mae = total_l1_loss / len(dataset)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)

    network.train()
    return mae, rmse, r2

def get_all_predictions(dataset, network, device):

    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    network.eval() 

    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    network.train()
    return np.concatenate(all_labels).ravel(), np.concatenate(all_preds).ravel()

def plot_regression_result(all_labels, all_preds, r2, mae, save_path='regression_scatter.png'):

    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_preds, alpha=0.5, s=10, c='#1f77b4', label='Predictions')
    max_val = max(all_labels.max(), all_preds.max())
    min_val = min(all_labels.min(), all_preds.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal (y=x)')
    range_val = max_val - min_val
    plt.text(min_val + 0.05 * range_val, max_val - 0.1 * range_val, 
             f'R2: {r2:.4f}\nMAE: {mae:.4f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.xlabel('Ground Truth (m/s)')
    plt.ylabel('Model Prediction (m/s)')
    plt.title('Typhoon Intensity Prediction')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(save_path) 
    plt.close()
    print(f"Regression scatter plot saved to {save_path}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    losses_his = [[], []]
    rmse_his = []
    r2_his = []
    best_r2 = float('-inf')  
    patience = 10         
    patience_counter = 0
    path_ = os.path.abspath('.')
    traindata = ImageFolder(os.path.join(path_, 'trainset'), train_transform)


    torch.manual_seed(42) 
    trainset, _ = random_split(traindata, [16000, len(traindata) - 16000])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    test_val_data = ImageFolder(os.path.join(path_, 'test_val_set'), test_val_transform)
    torch.manual_seed(99)
    valset, testset,_ = random_split(test_val_data, [2000, 2000,  len(test_val_data) - 4000])
    net = Net().to(device)

    loss_function = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9) 

    for epoch in range(50):
        print(f'Epoch {epoch + 1}',flush=True)
        running_loss = 0.0

 
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()  
            optimizer.zero_grad()  
            outputs = net(inputs)  
            loss = loss_function(outputs, labels) 
            loss.backward() 
            optimizer.step()  

            running_loss += loss.item()  


        avg_train_loss = running_loss / len(trainloader)
        avg_val_mae, avg_val_rmse, avg_val_r2 = validate(valset, net, device)
        log_to_csv('experiment_log.csv', epoch + 1, avg_train_loss, avg_val_mae, avg_val_rmse, avg_val_r2, optimizer.param_groups[0]['lr'])
        losses_his[0].append(avg_train_loss)
        losses_his[1].append(avg_val_mae)
        rmse_his.append(avg_val_rmse) 
        r2_his.append(avg_val_r2)     
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_mae:.4f}, RMSE: {avg_val_rmse:.4f}, R2: {avg_val_r2:.4f}', flush=True)
        scheduler.step()  
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

        if avg_val_r2 > best_r2:
            best_r2 = avg_val_r2
            patience_counter = 0  
            torch.save(net.state_dict(), os.path.join(path_, 'best_resnet_model.pth'))
            print(f"--- Best Model Saved (R2: {best_r2:.4f}) ---", flush=True)
        else:
            patience_counter += 1
            print(f"--- No improvement for {patience_counter} epochs ---", flush=True)
        
        if patience_counter >= patience:
            print("Early stopping triggered.", flush=True)
            break 
    print('Finished Training', flush=True)

    print("Loading Best Model for final evaluation...", flush=True)
    net.load_state_dict(torch.load(os.path.join(path_, 'best_resnet_model.pth'), weights_only=True))
    
    final_mae, final_rmse, final_r2 = validate(testset, net, device)
    y_true, y_pred = get_all_predictions(testset, net, device)
    plot_regression_result(y_true, y_pred, final_r2, final_mae, save_path='final_regression_scatter.png')
    print("="*40)
    print(f"FINAL TEST RESULTS (Best Model):")
    print(f"MAE: {final_mae:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"R2 Score: {final_r2:.4f}")
    print("="*40)

    epochs_range = range(1, len(losses_his[0]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, losses_his[0], label='Train L1 Loss')
    plt.plot(epochs_range, losses_his[1], label='Val MAE')
    plt.plot(epochs_range, rmse_his, label='Val RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Value')
    plt.title('Training and Validation Errors')
    plt.legend()
    plt.savefig('metrics_errors.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, r2_his, label='Val R2 Score', marker='o', color='green')

    plt.ylim(0, 1.0) 


    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.title('Model R2 Score Trend ')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig('metrics_r2_optimized.png')
    plt.close()
