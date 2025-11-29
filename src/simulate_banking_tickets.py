import pandas as pd
import numpy as np


CATEGORIES = [
    "card_lost_stolen",
    "transaction_dispute",
    "account_login_issue",
    "loan_inquiry",
    "account_update",
    "general_info"
]


def generate_examples(n_per_class=200, seed=42):
    np.random.seed(seed)
    rows = []

    templates = {
        "card_lost_stolen": [
            "I lost my debit card yesterday, please block it.",
            "My credit card was stolen and I need a replacement.",
            "Someone took my card, urgent block request."
        ],
        "transaction_dispute": [
            "There is an unknown charge on my account.",
            "I want to dispute a transaction from {merchant}.",
            "A payment I did not authorize appears on my card."
        ],
        "account_login_issue": [
            "I can't log into my online banking.",
            "My app says incorrect password but it's right.",
            "Two-factor code is not working for my login."
        ],
        "loan_inquiry": [
            "I want to know the interest rate for personal loans.",
            "How can I apply for a home loan?",
            "What documents are required for car loan approval?"
        ],
        "account_update": [
            "I need to update my address.",
            "How do I change my phone number on file?",
            "Please help me update my email for notifications."
        ],
        "general_info": [
            "What are your branch operating hours?",
            "Do you offer student accounts?",
            "Is there a fee for international transfers?"
        ]
    }

    for label, phrases in templates.items():
        for _ in range(n_per_class):
            text = np.random.choice(phrases)
            text = text.replace("{merchant}", np.random.choice(["Amazon", "Uber", "Netflix"]))
            rows.append({"text": text, "label": label})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_examples()
    df.to_csv(r"D:\Projects\Finance\NLP_for_Banking\data\banking_tickets.csv", index=False)
    print(df.head())
