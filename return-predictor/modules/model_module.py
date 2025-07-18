from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
import numpy as np
import os

# Architecture: layers of Linear->BatchNorm->ReLU activation->Dropout, input-256-128-64-32-output
# Using Torch's AdamW optimization, MSE loss, and a standard learning rate scheduler
class StockNN(nn.Module):
    def __init__(self, input_dim=12, dropout_rate=0.25):
        super().__init__()

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        self.layers = nn.Sequential(
            block(input_dim, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.layers(x)
    

class CustomTabularDataset(Dataset):
    def __init__(self, dataframe, 
                 target_columns=["lbl_next_three_days_ret", "lbl_next_week_ret", "lbl_next_two_weeks_ret", "lbl_next_month_ret", "lbl_next_two_months_ret"],
                 return_targets=True):
        self.return_targets = return_targets and all(col in dataframe.columns for col in target_columns)
        
        target_columns_and_index = target_columns + ["ticker_and_date"]
        drop_cols = [col for col in target_columns_and_index if col in dataframe.columns]
        
        self.dataframe = dataframe
        self.features = torch.tensor(dataframe.drop(columns=drop_cols).values, dtype=torch.float32)

        if self.return_targets:
            self.targets = torch.tensor(dataframe[target_columns].values, dtype=torch.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.return_targets:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]

class Model:
    def __init__(self, epochs=100, batch_size=25):
        self.num_epochs = epochs
        self.batch_size = batch_size

        mps_device = None
        if torch.backends.mps.is_available():
            print("Activated MPS for training, because you're on a Mac.")
            mps_device = torch.device("mps")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not mps_device else mps_device
        self.model = StockNN().to(self.device)

    def create_dataloader_for_pred(self, df):
        dataset = CustomTabularDataset(df, return_targets=False)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def load(self):
        filename = "results/saved_model.pth"
        if os.path.isfile(filename):
            self.model.load_state_dict(torch.load(filename, weights_only=True, map_location=torch.device('cpu')))
            self.model.eval()
            return
        raise Exception(f"Model is not found on path {filename}!")
    
    def test(self, dataloader: DataLoader, model=None):
        if model is None:
            model = self.model
        model.eval()
        all_preds = []

        with torch.no_grad():
            for x in tqdm(dataloader, unit="batch"):
                x = x.float().to(self.device)
                pred = model(x)
                all_preds.append(pred.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        return all_preds
    
    def train(self, df, df_valid):
        model = self.model

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        dataset = CustomTabularDataset(df)
        dataset_valid = CustomTabularDataset(df_valid)

        da_valid, da_test = random_split(dataset_valid, [0.7, 0.3], torch.Generator().manual_seed(100))
        dl_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dl_valid = DataLoader(da_valid, batch_size=self.batch_size, shuffle=False)
        dl_test = DataLoader(da_test, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_model_path = "results/saved_model.pth"

        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch}...")
            model.train()
            for batch_x, batch_y in tqdm(dl_train, unit="batch"):
                x, y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            losses = []
            with torch.no_grad():
                for x, y in tqdm(dl_valid, unit="batch"):
                    x, y = x.float().to(self.device), y.float().to(self.device)
                    pred = model(x)
                    l = criterion(pred, y).item()
                    losses.append(l)
                        
            epoch_validation_loss = np.mean(losses)

            print(f"Finished epoch {epoch + 1} with loss {epoch_validation_loss}")
        
            if epoch_validation_loss < best_val_loss:
                best_val_loss = epoch_validation_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved at epoch {epoch + 1} with validation loss {epoch_validation_loss:.4f}")

        print("Evaluating on test set with best model...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, y in tqdm(dl_test, unit="batch"):
                x, y = x.float().to(self.device), y.float().to(self.device)
                pred = model(x)
                l = criterion(pred, y).item()
                test_losses.append(l)
        print(f"Test Loss: {np.mean(test_losses):.4f}")
        print(f"Model is at {best_model_path}, val loss at {best_val_loss}, test loss at {np.mean(test_losses)}")