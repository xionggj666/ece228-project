from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from DeapDataset import *
from models import *


class Pipeline:
    def __init__(self, model, train_data, train_label, test_data, test_label, num_classes=10, batch_size=32, lr=1e-3, epochs=200, mode="2D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = DEAPDataset(train_data, train_label, mode)
        self.test_dataset = DEAPDataset(test_data, test_label, mode)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels, total_loss = [], [], 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y.argmax(dim=1))
                preds = output.argmax(dim=1)
                true = y.argmax(dim=1)
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(true.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(loader)
        return acc, avg_loss

    def train(self):
        train_accs, train_losses = [], []
        test_accs, test_losses = [], []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch} Training"):
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y.argmax(dim=1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                preds = output.argmax(dim=1)
                true = y.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(true.cpu().numpy())

            train_acc = accuracy_score(all_labels, all_preds)
            train_loss = total_loss / len(self.train_loader)
            test_acc, test_loss = self.evaluate(self.test_loader)

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Plot results
        epochs = np.arange(1, self.epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label="Train Acc")
        plt.plot(epochs, test_accs, label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()