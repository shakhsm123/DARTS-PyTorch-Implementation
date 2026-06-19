import os
import torch
import torch.nn.functional as F
from network_search import Search_Network
from architect import Architect_second_order
from data import get_search_loaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('checkpoints', exist_ok=True)

train_loader, val_loader = get_search_loaders(batch_size=64)

model = Search_Network(C_init=16, num_classes=10).to(device)
architect = Architect_second_order(model, arch_lr=3e-4, arch_weight_decay=1e-3)
weight_params = [p for p in model.parameters() if p not in set(model.arch_parameters())]
optimizer = torch.optim.SGD(weight_params, lr=0.025, momentum=0.9, weight_decay=3e-4)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

for epoch in range(50):
    total_loss = 0
    count = 0

    for (batch_val, label_val), (batch_train, label_train) in zip(val_loader, train_loader):
        batch_train = batch_train.to(device)
        label_train = label_train.to(device)
        batch_val   = batch_val.to(device)
        label_val   = label_val.to(device)

        train_result = model(batch_train)
        loss = F.cross_entropy(train_result, label_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        architect.step(batch_val, label_val, batch_train, label_train, xi=0.025)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    cosine_scheduler.step()
    print(f"Epoch {epoch} | Loss {avg_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': cosine_scheduler.state_dict(),
            'arch_normal': model.alpha_normal.data,
            'arch_reduce': model.alpha_reduce.data,
            'avg_loss': avg_loss,
        }, f'checkpoints/search_epoch_{epoch}.pt')
        print(f"checkpoint saved at epoch {epoch}")
