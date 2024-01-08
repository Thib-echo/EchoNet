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
import pandas as pd
from pathlib import Path

def create_dataloader(input_ids, attention_masks, labels, batch_size=16):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def calculate_class_weights(labels):
    # Ensure labels are numpy array for processing
    labels_np = np.array(labels)
    # Compute class weights
    unique_classes = np.unique(labels_np)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels_np)
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

    # Calculate and log metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    writer.add_scalar(f'{phase} Loss', avg_loss, epoch)
    writer.add_scalar(f'{phase} Accuracy', avg_accuracy, epoch)
    writer.add_scalar(f'{phase} Precision', precision, epoch)
    writer.add_scalar(f'{phase} Recall', recall, epoch)
    writer.add_scalar(f'{phase} F1 Score', f1, epoch)
    print(f"Epoch {epoch+1}, {phase} Loss: {avg_loss}, {phase} Accuracy: {avg_accuracy}, {phase} Precision: {precision}, {phase} Recall: {recall}, {phase} F1: {f1}")

    # Return metrics along with predictions and labels for further analysis
    return avg_loss, avg_accuracy, precision, recall, f1, all_labels, all_preds



def run_experiment(model, train_loader, validation_loader, num_epochs, lr, device, writer, train_labels):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_weights = calculate_class_weights(train_labels)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion, device, writer, epoch)
        train_loss, train_acc, *_ = evaluate(model, train_loader, criterion, device, writer, epoch, 'Train')
        val_loss, val_acc, val_precision, val_recall, val_f1, *_ = evaluate(model, validation_loader, criterion, device, writer, epoch, 'Validation')

def save_dataset(df, file_name):
    df.to_csv(file_name, index=False)


def main():
    num_epochs = 1
    lr = 2e-5
    labels_file = Path("./sampled_data.csv")
    base_output_path = Path(f'outputs/{labels_file.stem}')
    base_runs_path = Path(f'runs/{labels_file.stem}')

    # Determine the next experiment number
    exp_number = 0
    while (base_output_path / f"exp_{exp_number}").exists() or (base_runs_path / f"exp_{exp_number}").exists():
        exp_number += 1

    # Create specific paths for this experiment
    output_path = base_output_path / f"exp_{exp_number}"
    runs_path = base_runs_path / f"exp_{exp_number}"

    writer = SummaryWriter(str(runs_path))
    train_dataset_path = Path(f'datasets/{labels_file.stem}/train_dataset.csv')
    validation_dataset_path = Path(f'datasets/{labels_file.stem}/validation_dataset.csv')
    test_dataset_path = Path(f'datasets/{labels_file.stem}/test_dataset.csv')

    if not (train_dataset_path.exists() and validation_dataset_path.exists() and test_dataset_path.exists()):

        transcripts, urgency_levels = load_transcripts_and_labels(labels_file)
        label_mapping, inverse_label_mapping = create_label_mapping(urgency_levels)
        transcripts = preprocess_transcripts(transcripts)

        full_dataset = pd.DataFrame({
            'transcripts': transcripts,
            'labels': [label_mapping[label] for label in urgency_levels]
        })

        # Splitting the data into training, validation, and test sets
        train_df, temp_df = train_test_split(full_dataset, random_state=2018, test_size=0.2)
        validation_df, test_df = train_test_split(temp_df, random_state=2018, test_size=0.5)

        # Save the datasets
        train_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(train_dataset_path, index=False)
        validation_df.to_csv(validation_dataset_path, index=False)
        test_df.to_csv(test_dataset_path, index=False)
    else:
        train_df = pd.read_csv(train_dataset_path, encoding='utf-8')
        validation_df = pd.read_csv(validation_dataset_path, encoding='utf-8')
        test_df = pd.read_csv(test_dataset_path, encoding='utf-8')

    # Process and create DataLoaders for the model
    train_inputs, train_masks, train_labels = preprocess_data(train_df['transcripts'], train_df['labels'])
    validation_inputs, validation_masks, validation_labels = preprocess_data(validation_df['transcripts'], validation_df['labels'])
    test_inputs, test_masks, test_labels = preprocess_data(test_df['transcripts'], test_df['labels'])

    train_loader = create_dataloader(train_inputs, train_masks, train_labels)
    validation_loader = create_dataloader(validation_inputs, validation_masks, validation_labels)
    test_loader = create_dataloader(test_inputs, test_masks, test_labels)

    model = BERTClassifier(num_classes=len(torch.unique(train_labels.clone().detach())))
    model, device = move_model_to_gpu(model)

    run_experiment(model, train_loader, validation_loader, num_epochs, lr, device, writer, train_labels)

    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "bert_transcript_classifier.pth"
    torch.save(model.state_dict(), model_path)

    # Evaluate the model on the test set
    print("Evaluating on test set")
    *_, true_labels, predictions = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, writer, -1, 'Test')

    # Save predictions and true labels
    torch.save({'predictions': predictions, 'true_labels': true_labels}, output_path / 'model_predictions.pt')

if __name__ == '__main__':
    main()