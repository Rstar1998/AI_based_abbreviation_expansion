from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import json


# Dataset class for fine-tuning
class AbbreviationDataset(Dataset):
    def __init__(self, sentences, abbreviations, expansions, tokenizer, max_len=128):
        self.sentences = sentences
        self.abbreviations = abbreviations
        self.expansions = expansions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx].replace(self.abbreviations[idx], "[MASK]")
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)
        label = self.tokenizer(self.expansions[idx], return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)
        inputs['labels'] = label['input_ids'].squeeze()
        return {key: val.squeeze() for key, val in inputs.items()}
    

# Function to fine-tune the model
def fine_tune_model(sentences, abbreviations, expansions, model_name='bert-base-uncased', epochs=3, batch_size=16, lr=5e-5):
    """
    Fine-tune the BERT model for abbreviation expansion.

    Args:
        sentences (list): List of input sentences.
        abbreviations (list): List of abbreviations in the sentences.
        expansions (list): List of correct expansions for the abbreviations.
        model_name (str): Pretrained model name (default: 'bert-base-uncased').
        epochs (int): Number of fine-tuning epochs (default: 3).
        batch_size (int): Batch size for training (default: 16).
        lr (float): Learning rate (default: 5e-5).

    Returns:
        AutoModelForMaskedLM: The fine-tuned model.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.train()

    # Prepare dataset and dataloader
    dataset = AbbreviationDataset(sentences, abbreviations, expansions, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Fine-tuning loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

    return model

def predict_with_fine_tuned_model(sentence: str, abbreviation: str, model_path: str = "fine_tuned_abbreviation_model"):
    """
    Use the fine-tuned model and tokenizer to predict abbreviation expansion.

    Args:
        sentence (str): The sentence containing the abbreviation.
        abbreviation (str): The abbreviation to expand.
        model_path (str): Path to the directory containing the fine-tuned model and tokenizer.

    Returns:
        str: The predicted expansion for the abbreviation.
    """
    try:
        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)

        # Replace the abbreviation with [MASK]
        masked_sentence = sentence.replace(abbreviation, "[MASK]")

        # Tokenize Input
        inputs = tokenizer(masked_sentence, return_tensors="pt")

        # Get Predictions
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits

        # Find the [MASK] token and its predicted token
        masked_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        predicted_token_id = predictions[0, masked_index].argmax(dim=1)
        predicted_token = tokenizer.decode(predicted_token_id)

        return predicted_token.strip()
    except Exception as e:
        return f"Error during prediction: {str(e)}"



if __name__ == "__main__":

    # Load the dataset
    with open('abbreviation_dataset.json', 'r') as file:
        dataset = json.load(file)

    # Prepare sentences, abbreviations, and expansions
    sentences = [item['sentence'] for item in dataset]
    abbreviations = [item['abbreviation'] for item in dataset]
    expansions = [item['expansion'] for item in dataset]

    print("Sentences:", sentences)
    print("Abbreviations:", abbreviations)
    print("Expansions:", expansions)

    model = 'bert-base-uncased'

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(sentences, abbreviations, expansions, model)

    # Save the model and tokenizer
    fine_tuned_model.save_pretrained("fine_tuned_abbreviation_model")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.save_pretrained("fine_tuned_abbreviation_model")




    # # Example predictions
    # test_sentence = "The VIP section was reserved for special guests."
    # test_abbreviation = "VIP"
    # predicted_expansion = predict_with_fine_tuned_model(test_sentence, test_abbreviation)
    # print(f"Predicted Expansion for '{test_abbreviation}': {predicted_expansion}")

    # test_sentence_2 = "The WHO released new guidelines on pandemic control."
    # test_abbreviation_2 = "WHO"
    # predicted_expansion_2 = predict_with_fine_tuned_model(test_sentence_2, test_abbreviation_2)
    # print(f"Predicted Expansion for '{test_abbreviation_2}': {predicted_expansion_2}")
