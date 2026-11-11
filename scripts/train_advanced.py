import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import CharTokenizer
from src.amn4w import AMN4W
from src.transformer_baseline import DecoderOnlyTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        if isinstance(context, list):
            context = " ".join(context)
        question = item["question"]
        answer = item["answer"]
        text = f"{context} Soru: {question} Cevap: {answer}"
        tokens = self.tokenizer.encode(text, max_len=self.max_len+1)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if isinstance(model, AMN4W):
            logits, _ = model(x)
        else:
            logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--memory_size", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    tokenizer = CharTokenizer()
    with open("data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("data/test_new_knowledge.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    texts = []
    for item in train_data + test_data:
        context = item["context"] if not isinstance(item["context"], list) else " ".join(item["context"])
        texts.append(f"{context} Soru: {item['question']} Cevap: {item['answer']}")
    tokenizer.fit(texts)

    os.makedirs("weights", exist_ok=True)
    tokenizer.save(f"weights/tokenizer_{args.model}_{args.seed}.json")

    train_dataset = QADataset("data/train.json", tokenizer, args.max_len)
    if len(train_dataset) == 0:
        return
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model == "amn4w":
        model = AMN4W(tokenizer.vocab_size, args.embed_dim, args.memory_size, args.max_len)
    elif args.model == "transformer":
        model = DecoderOnlyTransformer(tokenizer.vocab_size, args.embed_dim, 2, 2, args.max_len)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    log_file = f"results/{args.model}_adv_seed{args.seed}_log.json"
    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % 20 == 0:
            print(f"[{args.model}] Seed {args.seed} - Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}")
        history.append({"epoch": epoch, "loss": loss})

    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), f"weights/{args.model}_adv_seed{args.seed}.pt")

if __name__ == "__main__":
    main()
