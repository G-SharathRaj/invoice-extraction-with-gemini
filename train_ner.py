import spacy
from spacy.training.example import Example
import json
import os

# Load training data (modify according to your dataset)
TRAIN_DATA = [
    ("Invoice No: 12345 Date: 12/05/2024 Customer: John Doe GST: 22AAACC1234F1Z5 Business: Foodies Total: $500.00",
     {"entities": [(12, 17, "INVOICE_NO"), (24, 34, "DATE"), (45, 53, "CUSTOMER_NAME"),
                   (58, 72, "GST"), (83, 91, "BUSINESS_NAME"), (100, 107, "TOTAL")]}),
]

# Load blank spaCy model
nlp = spacy.blank("en")

# Create Named Entity Recognizer
ner = nlp.add_pipe("ner", last=True)

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Training
optimizer = nlp.begin_training()
for epoch in range(10):  # Train for 10 epochs
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses, drop=0.5)
    print(f"Losses at epoch {epoch}: {losses}")

# Save model
os.makedirs("models/invoice_ner_model", exist_ok=True)
nlp.to_disk("models/invoice_ner_model")
print("✅ Model Training Completed & Saved!")
