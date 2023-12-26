import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from model import BERTClassifier, move_model_to_gpu
from utils import load_transcripts_and_labels, preprocess_data, preprocess_transcripts, create_label_mapping
from sklearn.model_selection import train_test_split

def create_dataloader(input_ids, attention_masks, labels, batch_size=16):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for step, (input_ids, masks, labels) in enumerate(dataloader):
            input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} finished.")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    
    for step, (input_ids, masks, labels) in enumerate(dataloader):
        input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_accuracy += (preds == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader.dataset)
    return avg_loss, avg_accuracy

def run_experiment(model, train_loader, validation_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    train(model, train_loader, optimizer, criterion, device)
    train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
    val_loss, val_accuracy = evaluate(model, validation_loader, criterion, device)

    return train_loss, train_accuracy, val_loss, val_accuracy

def main():
    dummy_data_dir = "./processed_files/"

    transcripts, urgency_levels = load_transcripts_and_labels(dummy_data_dir, './dummy_labels.csv')

    # Create label mappings
    label_mapping, inverse_label_mapping = create_label_mapping(urgency_levels)

    transcripts = preprocess_transcripts(transcripts)
    input_ids, attention_masks, labels = preprocess_data(transcripts,  [label_mapping[label] for label in urgency_levels])
    
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    train_loader = create_dataloader(train_inputs, train_masks, train_labels)
    validation_loader = create_dataloader(validation_inputs, validation_masks, validation_labels)

    model = BERTClassifier(num_classes=len(torch.unique(labels)))
    model, device = move_model_to_gpu(model)

    # Run experiment
    train_loss, train_acc, val_loss, val_acc = run_experiment(model, train_loader, validation_loader, device)
    print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

    # Save the trained model
    model_path = 'bert_transcript_classifier.pth'
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()