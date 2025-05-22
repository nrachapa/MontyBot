import torch
from net import SimpleNNUE 
import numpy as np

def train():
    model = SimpleNNUE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFunc = torch.nn.MSELoss()

    X = torch.tensor(np.random.rand(1000, 768).astype(np.float32))
    y = torch.tensor(np.random.rand(1000, 1).astype(np.float32))

    for epoch in range(10):
        pred = model(X)
        loss = lossFunc(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()