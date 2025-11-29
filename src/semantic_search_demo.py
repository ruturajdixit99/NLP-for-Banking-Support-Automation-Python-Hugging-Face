import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import faiss


def embed_texts(texts, tokenizer, model, device, max_len=64):
    model.eval()
    all_emb = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]   # CLS token
            all_emb.append(cls_emb.cpu().numpy())
    return np.vstack(all_emb)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    faqs = pd.DataFrame({
        "question": [
            "How do I block a lost card?",
            "How can I dispute a transaction?",
            "How do I reset my online banking password?",
            "What is your home loan interest rate?",
        ],
        "answer": [
            "You can block your card immediately via the mobile app under 'Cards > Block card'.",
            "Go to 'Transactions', select the transaction, and click 'Dispute'.",
            "Use 'Forgot password' on the login page and follow the verification steps.",
            "Our current home loan rates start from 7.5% p.a., subject to eligibility.",
        ]
    })

    faq_emb = embed_texts(faqs["question"].tolist(), tok, encoder, device)
    dim = faq_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(faq_emb)
    index.add(faq_emb)

    user_query = "I lost my debit card what should I do"
    q_emb = embed_texts([user_query], tok, encoder, device)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=2)

    print("Query:", user_query)
    print("Top FAQ match:", faqs.iloc[I[0][0]]["question"])
    print("Answer:", faqs.iloc[I[0][0]]["answer"])
