import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class AutoEncoderTrainer:
    def __init__(self, cfg):
        self.latent_dim = cfg.get("latent_dim", 64)
        self.batch_size = cfg.get("batch_size", 512)
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 20)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = cfg["feature_cols"]  # list of numeric columns

    def _prepare_data(self, df):
        X = df[self.feature_cols].to_numpy().astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader, X.shape[1]

    def fit(self, df_train):
        loader, input_dim = self._prepare_data(df_train)
        model = SimpleAE(input_dim, self.latent_dim).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optim.zero_grad()
                x_hat, _ = model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optim.step()
                total_loss += loss.item() * batch.size(0)
            avg_loss = total_loss / len(loader.dataset)
            print(f"[AE] Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.6f}")

        meta = {
            "input_dim": input_dim,
            "latent_dim": self.latent_dim,
            "feature_cols": self.feature_cols,
        }
        return model, meta

    @staticmethod
    def encode(model, df, feature_cols):
        model.eval()
        with torch.no_grad():
            X = df[feature_cols].to_numpy().astype(np.float32)
            X_t = torch.from_numpy(X)
            if next(model.parameters()).is_cuda:
                X_t = X_t.cuda()
            _, Z = model(X_t)
            Z_np = Z.cpu().numpy()
        return Z_np
