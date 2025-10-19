# generate.py
import torch
from model import MiniGPT
from tokenizer import CharTokenizer

text = open("data/input.txt", "r").read()
tokenizer = CharTokenizer(text)

model = MiniGPT(len(tokenizer.stoi))
model.load_state_dict(torch.load("miniGPT.pth", map_location="cpu"))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long)
out = context

for _ in range(200):
    logits, _ = model(out[:, -128:])
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    out = torch.cat((out, next_token), dim=1)

print(tokenizer.decode(out[0].tolist()))

