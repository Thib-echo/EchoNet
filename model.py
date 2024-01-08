import torch.nn as nn
from transformers import CamembertModel, FlaubertModel
from transformers import CamembertTokenizer, FlaubertTokenizer

class BERTClassifier(nn.Module):
    def __init__(self, transformer_model, num_classes, tokenizer):
        super(BERTClassifier, self).__init__()
        self.transformer = transformer_model
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        return self.fc(dropout_output)
    
    def tokenize(self, texts, max_length=512):
        encoded_data = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoded_data['input_ids'], encoded_data['attention_mask']
    
class CamembertClassifier(BERTClassifier):
    def __init__(self, num_classes, model_name='camembert-base'):
        camembert = CamembertModel.from_pretrained(model_name)
        tokenizer = CamembertTokenizer.from_pretrained(model_name)
        super().__init__(camembert, num_classes, tokenizer)

class FlaubertClassifier(BERTClassifier):
    def __init__(self, num_classes, model_name='flaubert-base-uncased'):
        flaubert = FlaubertModel.from_pretrained(model_name)
        tokenizer = FlaubertTokenizer.from_pretrained(model_name)
        super().__init__(flaubert, num_classes, tokenizer)