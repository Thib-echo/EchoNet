import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import CamembertTokenizer
import re
import string
import torch

def load_transcripts_and_labels(labels_file):
    labels_df = pd.read_csv(labels_file, encoding='utf-8')
    transcripts = labels_df['transcript'].tolist()
    labels = labels_df['Devenir'].tolist()
    return transcripts, labels

def preprocess_data(transcripts, labels):
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    encoded_data = tokenizer.batch_encode_plus(
        transcripts,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_data['input_ids'], encoded_data['attention_mask'], torch.tensor(encoded_labels)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZéèêëîïôœûüçàáâäæçÉÈÊËÎÏÔŒÛÜÇÀÁÂÄÆ0-9]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_transcripts(transcripts):
    return [clean_text(transcript) for transcript in transcripts]

def create_label_mapping(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return label_mapping, inverse_label_mapping