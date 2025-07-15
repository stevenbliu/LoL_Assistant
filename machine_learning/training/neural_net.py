# training/neural_net.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Union
import mlflow
import mlflow.pytorch
from training.logging import log_metrics_and_plots


class LSTMPositionPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

import mlflow
import mlflow.pytorch


class LSTMPositionPredictor(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, output_size=2, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])  # last layer's hidden state


import random
import numpy as np
import torch
from tqdm import tqdm  # pip install tqdm if you don't have it


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lstm_model(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="LSTM",
    n_epochs=20,
    batch_size=64,
    lr=1e-3,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    seed=42,
):
    print(f"\n🔁 Training {model_name} model...", flush=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 4  # 3 history + 1 current

    # Only reshape if input is a DataFrame
    if isinstance(X_train, pd.DataFrame):
        id_time_cols = ["MatchId", "P2_PlayerId", "Timestamp"]
        X_train_np = X_train.drop(columns=id_time_cols).values.astype(np.float32)
        X_test_np = X_test.drop(columns=id_time_cols).values.astype(np.float32)

        if X_train_np.shape[1] % total_timesteps != 0:
            raise ValueError(
                f"Feature count ({X_train_np.shape[1]}) not divisible by timesteps ({total_timesteps})"
            )

        input_dim = X_train_np.shape[1] // total_timesteps
        X_train_reshaped = X_train_np.reshape(-1, total_timesteps, input_dim)
        X_test_reshaped = X_test_np.reshape(-1, total_timesteps, input_dim)
    else:
        X_train_reshaped = X_train.astype(np.float32)
        X_test_reshaped = X_test.astype(np.float32)
        input_dim = X_train_reshaped.shape[2]

    y_train_np = (
        y_train.values.astype(np.float32)
        if isinstance(y_train, pd.DataFrame)
        else y_train.astype(np.float32)
    )
    y_test_np = (
        y_test.values.astype(np.float32)
        if isinstance(y_test, pd.DataFrame)
        else y_test.astype(np.float32)
    )

    output_dim = y_train_np.shape[1]

    print("X_train_reshaped shape:", X_train_reshaped.shape)
    print("y_train_np shape:", y_train_np.shape)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_reshaped), torch.tensor(y_train_np)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_reshaped), torch.tensor(y_test_np)),
        batch_size=batch_size,
    )

    model = LSTMPositionPredictor(
        input_size=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    mlflow.set_experiment("experiment")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(
            {
                "model": model_name,
                "epochs": n_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "seed": seed,
            }
        )

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for xb, yb in loop:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_reshaped).to(device)
            preds = model(X_test_tensor).cpu().numpy()

        log_metrics_and_plots(y_test_np, preds, model_name=model_name)

        mlflow.pytorch.log_model(model, artifact_path="model")
