import os
import torch
import torch.nn.functional as F
from network_search import Search_Network
from network_eval import Network_Eval
from genotype import get_genotype
from data import get_eval_loaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('checkpoints', exist_ok=True)

search_model = Search_Network(C_init=16, num_classes=10).to(device)
checkpoint = torch.load('checkpoints/search_epoch_49.pt')
search_model.load_state_dict(checkpoint['model_state_dict'])
arch = get_genotype(search_model)
print(arch)

train_loader, test_loader = get_eval_loaders(batch_size=96)

model = Network_Eval(C_init=36, num_classes=10, genotype=arch).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=0)

best_accuracy = 0.0

for epoch in range(600):
    drop_prob = 0.3 * epoch / 600
    for cell in model.cells:
        cell.drop_prob = drop_prob

    model.train()
    total_loss = 0
    count = 0

    for batch, labels in train_loader:
        batch  = batch.to(device)
        labels = labels.to(device)

        logits, aux_logits = model(batch)
        loss     = F.cross_entropy(logits, labels)
        aux_loss = F.cross_entropy(aux_logits, labels)
        loss_total = loss + 0.4 * aux_loss

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss_total.item()
        count += 1

    avg_loss = total_loss / count
    cosine_scheduler.step()

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch, labels in test_loader:
            batch  = batch.to(device)
            labels = labels.to(device)
            logits, _ = model(batch)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = 100 * correct / total
    test_error = 100 - accuracy
    print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | Test Acc {accuracy:.2f}% | Test Error {test_error:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': cosine_scheduler.state_dict(),
            'accuracy': accuracy,
        }, 'checkpoints/eval_best.pt')

    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': cosine_scheduler.state_dict(),
            'accuracy': accuracy,
        }, f'checkpoints/eval_epoch_{epoch}.pt')

print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")
print(f"Best Test Error:    {100 - best_accuracy:.2f}%")
