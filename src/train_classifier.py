import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


if __name__ == "__main__":
    df = pd.read_csv(r"D:\Projects\Finance\NLP_for_Banking\data\banking_tickets.csv")

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(le.classes_)
    )

    train_dataset = TicketDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer)
    test_dataset = TicketDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

    # evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    model.save_pretrained(r"D:\Projects\Finance\NLP_for_Banking\data\banking_classifier")
    tokenizer.save_pretrained(r"D:\Projects\Finance\NLP_for_Banking\data\banking_classifier")
    le.classes_.dump(r"D:\Projects\Finance\NLP_for_Banking\data\banking_label_classes.npy")
