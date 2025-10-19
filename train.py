# train.py
import torch
from model import MiniGPT
from tokenizer import CharTokenizer

# データ読み込み
text = open("data/input.txt", "r").read()
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# 学習設定
block_size = 8
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

model = MiniGPT(len(tokenizer.stoi)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(5000):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

torch.save(model.state_dict(), "miniGPT.pth")

