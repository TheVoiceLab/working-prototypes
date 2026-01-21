import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#A rather standard tokenizer, not embedding.
class StandardTokenizer:
    def __init__(self, sentences):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        token_counts = {}
        for s in sentences:
            for tok in s.strip().split():
                token_counts[tok] = token_counts.get(tok, 0) + 1

        self.itos = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for tok in token_counts:
            self.itos.append(tok)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.pad_id = self.stoi[self.pad_token]
        self.unk_id = self.stoi[self.unk_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]

    def encode(self, text, add_special_tokens=True):
        tokens = text.strip().split()
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == self.eos_id:
                break
            tok = self.itos[i]
            if tok in {self.bos_token, self.pad_token}:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    @property
    def vocab_size(self):
        return len(self.itos)

#we cut sentences to less than 16 words.
#fillin pad when length is less than 16
class LMDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=16):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer

        for s in sentences:
            ids = tokenizer.encode(s)
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        pad = self.tokenizer.pad_id

        input_ids = ids[:-1] + [pad] * (self.max_len - len(ids[:-1]))
        target_ids = ids[1:] + [pad] * (self.max_len - len(ids[1:]))

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

#Using a classic decoder model to generate suggestions
class TransformerLanguageModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=128,
            n_heads=4,
            num_layers=2,
            dim_feedforward=256,
            max_len=128,
            dropout=0.1,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.max_len = max_len

    def forward(self, x):
        bsz, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)

        h = self.token_emb(x) + self.pos_emb(pos)

        # Causal mask to prevent looking at the future
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        h = self.transformer(h, mask=mask)
        h = self.ln_f(h)
        return self.fc_out(h)


def train(model, dataloader, tokenizer, epochs=30, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | loss={total_loss / len(dataloader):.4f}")


# ============================================================
# 5. Generation + Confidence + EOS Detection
# ============================================================
def generate_with_confidence(
        model,
        tokenizer,
        prefix,
        max_new_tokens=20,
        temperature=1.0,
):
    model.eval()
    token_confidences = []
    generated_ids = []

    with torch.no_grad():
        # Encode prefix; manually handle special tokens to avoid double-BOS
        input_ids = tokenizer.encode(prefix, add_special_tokens=False)
        # Add BOS at the start manually
        input_ids = [tokenizer.bos_id] + input_ids

        x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        for _ in range(max_new_tokens):
            logits = model(x)
            # Take only the last token prediction
            next_token_logits = logits[0, -1, :] / temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            confidence, next_id = torch.max(probs, dim=-1)
            next_id = next_id.item()

            # --- EOS DETECTION ---
            if next_id == tokenizer.eos_id:
                break

            token_confidences.append(confidence.item())
            generated_ids.append(next_id)

            # Update context for next iteration
            x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)

            # Stop if context exceeds model max length
            if x.size(1) >= model.max_len:
                break

        decoded_leftover = tokenizer.decode(generated_ids)
        avg_conf = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0

        return {
            "full_text": prefix + " " + decoded_leftover,
            "leftover": decoded_leftover,
            "token_confidences": token_confidences,
            "average_confidence": avg_conf,
        }


def estimate_reading_time_ms(text, wpm=220):
    if not text.strip():
        return 0
    words = text.strip().split()
    word_count = len(words)
    ms = (word_count / wpm) * 60_000
    return int(ms)


# ============================================================
# 6. Save / Load
# ============================================================
def save_model(model, tokenizer, max_len, path="transformer_lm_1M_dup.pt"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.itos,
            "max_len": max_len,
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(path="transformer_lm_1M_dup.pt"):
    checkpoint = torch.load(path, map_location=DEVICE)

    tokenizer = StandardTokenizer([])
    tokenizer.itos = checkpoint["vocab"]
    tokenizer.stoi = {tok: i for i, tok in enumerate(tokenizer.itos)}
    tokenizer.pad_id = tokenizer.stoi["<pad>"]
    tokenizer.unk_id = tokenizer.stoi["<unk>"]
    tokenizer.bos_id = tokenizer.stoi["<bos>"]
    tokenizer.eos_id = tokenizer.stoi["<eos>"]

    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.itos),
        max_len=checkpoint["max_len"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded.")
    return model, tokenizer


# ============================================================
# 7. Main Execution
# ============================================================
if __name__ == "__main__":
    # 1. Dataset loading (Placeholder for your CSV)
    # sentences = ["the cat sat on the mat ."] # Default fallback

    try:
        source_loc = "."
        filename = source_loc + 'sample.csv'
        data = pd.read_csv(filename).values.tolist()[:100000]
        sentences = [str(item[0]) + ' .' for item in data]
        print(f'Dataset size: {len(sentences)}, sample: {sentences[0]}')
    except Exception as e:
        print(f"Could not load CSV, using default sentences. Error: {e}")
        sentences = [
            "the cat sat on the mat .",
            "the dog sat on the log .",
            "machine learning is fun .",
            "transformers are powerful models ."
        ]

    tokenizer = StandardTokenizer(sentences)
    max_len = 16

    # 2. Setup Model
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_len=max_len,
    ).to(DEVICE)

    # UNCOMMENT TO TRAIN:
    # dataset = LMDataset(sentences, tokenizer, max_len=max_len)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # train(model, dataloader, tokenizer, epochs=10)
    # save_model(model, tokenizer, max_len)

    # 3. Load existing model
    try:
        model, tokenizer = load_model()
    except:
        print("Pre-trained model not found. Please train first.")

    # 4. Interaction Loop
    print("\n--- Autocomplete System Ready ---")
    while True:
        prefix = input("\nEnter a prefix (or 'quit'): ").strip()
        if prefix.lower() == "quit":
            break

        for i in range(1):
            out = generate_with_confidence(
                model,
                tokenizer,
                prefix,
                max_new_tokens=10,
                temperature=0.2
            )

            leftover = out["leftover"]
            leftovertime = estimate_reading_time_ms(leftover, 225)

            print(f'=> {out["full_text"]}')
            print(f'   Confidence: {out["average_confidence"]:.2%}')
            print(f'   Expected streaming time to complete: {leftovertime}I ms')