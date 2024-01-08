import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from model import BERTClassifier, move_model_to_gpu
from utils import load_transcripts_and_labels, preprocess_data, preprocess_transcripts, create_label_mapping
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def create_dataloader(input_ids, attention_masks, labels, batch_size=16):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def calculate_class_weights(labels):
    # Compute class weights using sklearn's utility function
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def train(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for step, (input_ids, masks, labels) in enumerate(dataloader):
        input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + step)

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Average Training Loss', avg_loss, epoch)
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_loss}")

def evaluate(model, dataloader, criterion, device, writer, epoch, phase):
    model.eval()
    total_loss, total_accuracy = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for step, (input_ids, masks, labels) in enumerate(dataloader):
            input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=masks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_accuracy += (preds == labels).sum().item()

            # Store predictions and labels for calculating additional metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader.dataset)

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Log the metrics
    writer.add_scalar(f'{phase} Loss', avg_loss, epoch)
    writer.add_scalar(f'{phase} Accuracy', avg_accuracy, epoch)
    writer.add_scalar(f'{phase} Precision', precision, epoch)
    writer.add_scalar(f'{phase} Recall', recall, epoch)
    writer.add_scalar(f'{phase} F1 Score', f1, epoch)

    print(f"Epoch {epoch+1}, {phase} Loss: {avg_loss}, {phase} Accuracy: {avg_accuracy}, {phase} Precision: {precision}, {phase} Recall: {recall}, {phase} F1: {f1}")

    return avg_loss, avg_accuracy, precision, recall, f1


def run_experiment(model, train_loader, validation_loader, device, writer, train_labels):
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    class_weights = calculate_class_weights(train_labels)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(10):
        train(model, train_loader, optimizer, criterion, device, writer, epoch)
        train_loss, train_acc, _, _, _ = evaluate(model, train_loader, criterion, device, writer, epoch, 'Train')
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, validation_loader, criterion, device, writer, epoch, 'Validation')


def main():
    writer = SummaryWriter('runs/urgent_transcript_classifier')
    labels_file = "./sampled_data_1to_3.csv"

    transcripts, urgency_levels = load_transcripts_and_labels(labels_file)
    label_mapping, inverse_label_mapping = create_label_mapping(urgency_levels)
    transcripts = preprocess_transcripts(transcripts)
    input_ids, attention_masks, labels = preprocess_data(transcripts, [label_mapping[label] for label in urgency_levels])

    # Splitting the data into training, validation, and test sets
    train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.2)
    train_masks, temp_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.2)
    
    validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_inputs, temp_labels, random_state=2018, test_size=0.5)
    validation_masks, test_masks, _, _ = train_test_split(temp_masks, temp_inputs, random_state=2018, test_size=0.5)

    train_loader = create_dataloader(train_inputs, train_masks, train_labels)
    validation_loader = create_dataloader(validation_inputs, validation_masks, validation_labels)
    test_loader = create_dataloader(test_inputs, test_masks, test_labels)  # DataLoader for the test set

    model = BERTClassifier(num_classes=len(torch.unique(labels)))
    model, device = move_model_to_gpu(model)

    run_experiment(model, train_loader, validation_loader, device, writer, train_labels.numpy())

    model_path = 'models/bert_transcript_classifier.pth'
    torch.save(model.state_dict(), model_path)

    # Evaluate the model on the test set
    print("Evaluating on test set")
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, writer, -1, 'Test')

if __name__ == '__main__':
    main()