import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import re
import string
import torch

def load_transcripts_and_labels(data_dir, labels_file):
    data_dir = Path(data_dir)
    labels_df = pd.read_csv(labels_file)
    transcripts = []
    labels = []

    for row in labels_df.itertuples():
        transcript_file = data_dir / row.file_name / (row.file_name + '.txt')
        with open(transcript_file, 'r', encoding='utf-8') as file:
            transcript = file.read()
            transcripts.append(transcript)
            labels.append(row.urgency_level)

    return transcripts, labels

def preprocess_data(transcripts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
    text = text.lower()  # Lowercasing
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text

def preprocess_transcripts(transcripts):
    return [clean_text(transcript) for transcript in transcripts]

def create_label_mapping(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return label_mapping, inverse_label_mapping