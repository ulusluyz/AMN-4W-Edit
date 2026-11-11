import os
import json
import torch
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import CharTokenizer
from src.amn4w import AMN4W
from src.transformer_baseline import DecoderOnlyTransformer, KVBaseline

def load_model_and_tokenizer(model_type, seed, max_len, embed_dim, memory_size, device):
    tokenizer = CharTokenizer()
    tokenizer.load(f"weights/tokenizer_{model_type}_{seed}.json")

    if model_type == "amn4w":
        model = AMN4W(tokenizer.vocab_size, embed_dim, memory_size, max_len)
    elif model_type == "transformer":
        model = DecoderOnlyTransformer(tokenizer.vocab_size, embed_dim, 2, 2, max_len)
    elif model_type == "kvbaseline":
        model = KVBaseline(tokenizer.vocab_size, embed_dim, memory_size, max_len)

    model.load_state_dict(torch.load(f"weights/{model_type}_seed{seed}.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, context, question, max_len, device):
    if isinstance(context, list):
        context = " ".join(context)
    prompt = f"{context} Soru: {question} Cevap: "

    tokens = tokenizer.encode(prompt, max_len=None)
    tokens = [t for t in tokens if t not in [0, 3]]

    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    generated = []
    with torch.no_grad():
        for _ in range(8): # Answer lengths are short (e.g. mavi, sarı)
            if isinstance(model, AMN4W):
                logits, _ = model(input_ids)
            else:
                logits = model(input_ids)

            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            if next_token == tokenizer.char2idx.get("<eos>", 3):
                break

            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated).strip()

def evaluate_dataset(data_path, model, tokenizer, max_len, device):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return 0, []

    correct = 0
    total = len(data)
    results = []
    for item in data:
        pred_ans = generate_answer(model, tokenizer, item["context"], item["question"], max_len, device)
        is_correct = item["answer"].lower() in pred_ans.lower()
        if is_correct:
            correct += 1
        results.append({
            "question": item["question"],
            "expected": item["answer"],
            "predicted": pred_ans,
            "correct": is_correct
        })
    acc = correct / total if total > 0 else 0
    return acc, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--memory_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, args.seed, args.max_len, args.embed_dim, args.memory_size, device)

    datasets = {
        "train_eval": "data/train.json",
        "new_knowledge": "data/test_new_knowledge.json",
        "update": "data/test_update.json"
    }

    final_report = {}
    for name, path in datasets.items():
        acc, res = evaluate_dataset(path, model, tokenizer, args.max_len, device)
        final_report[name] = {"accuracy": acc, "details": res}
        print(f"[{args.model}] Seed {args.seed} - {name} Accuracy: {acc*100:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_{args.model}_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
