"""Training script for Pokemon FastGen Flow Matching models."""

import os
from pathlib import Path
import torch
from torch import optim
from tqdm import tqdm

from dataset import PokemonDataModule
from model import ModelOneNFE, ModelFewNFE


def train_model(model, dataloader, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            if isinstance(batch, (tuple, list)):
                images, categories = batch
            else:
                images, categories = batch, None

            images = images.to(device)
            if categories is not None:
                categories = categories.to(device)

            batch_size = images.shape[0]
            
            # Flow Matching Doğrusal İnterpolasyon
            x_0 = images
            x_1 = torch.randn_like(x_0)
            
            t = torch.rand(batch_size, device=device)
            t_expand = t[:, None, None, None]
            x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
            target_v = x_1 - x_0

            optimizer.zero_grad(set_to_none=True)

            # MPS üzerinde standart float32 ile kararlı ileri ve geri yayılım
            pred_v = model.forward(x_t, t, category=categories)
            loss = torch.mean((pred_v - target_v) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} Ortalama Loss: {total_loss / len(dataloader):.4f}")


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Eğitim Cihazı (Device): {device}")

    data_module = PokemonDataModule(
        data_root="./data/pokemon-generation-one-22k",
        split_dir="./data/pokemon-generation-one-22k",
        batch_size=32,
        num_workers=4,
        return_category=True
    )
    train_loader = data_module.train_dataloader()

    os.makedirs("./checkpoints", exist_ok=True)

    # --- 1. ModelFewNFE Eğitimi ---
    print("\n--- ModelFewNFE Eğitimi Başlıyor ---")
    model_few = ModelFewNFE(device=device)
    optimizer_few = optim.AdamW(model_few.parameters(), lr=2e-4, weight_decay=1e-4)
    
    train_model(model_few, train_loader, optimizer_few, device, epochs=30)
    
    torch.save({"state_dict": model_few.state_dict()}, "./checkpoints/few_nfe.ckpt")
    print("ModelFewNFE kaydedildi: ./checkpoints/few_nfe.ckpt")

    # --- 2. ModelOneNFE Eğitimi ---
    print("\n--- ModelOneNFE Eğitimi Başlıyor ---")
    model_one = ModelOneNFE(device=device)
    optimizer_one = optim.AdamW(model_one.parameters(), lr=2e-4, weight_decay=1e-4)

    train_model(model_one, train_loader, optimizer_one, device, epochs=30)

    torch.save({"state_dict": model_one.state_dict()}, "./checkpoints/one_nfe.ckpt")
    print("ModelOneNFE kaydedildi: ./checkpoints/one_nfe.ckpt")


if __name__ == "__main__":
    main()