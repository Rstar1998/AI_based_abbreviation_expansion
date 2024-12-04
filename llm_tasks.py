#Importing the Required Python Packages
import string
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
import spacy

from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np

pd.set_option('display.max_colwidth', None)

# Load dataset if not on kaggle
# !wget -nc -P data/ https://zenodo.org/record/4482922/files/train.csv
# !wget -nc -P data/ https://zenodo.org/record/4482922/files/valid.csv
# !wget -nc -P data/ https://zenodo.org/record/4482922/files/test.csv



# Lets load the default english model of spacy
# !python -m spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')

# datapath = "../input/medal-emnlp/pretrain_subset"
datapath = "archive/pretrain_subset"
# datapath = "data"

# Lets load the train dataset.

train = pd.read_csv(datapath + '/train.csv')
# train = train[:100000]

# Lets load validation and test datasets as well
valid = pd.read_csv(f'{datapath}/valid.csv')
# test = pd.read_csv(f'{datapath}/test.csv')
# valid = valid[:10000]
# test = test[:1000]


# Lets create a function to create a new feature 'ABV' from dataset
def createFeature(df):    
    return [x.split(' ')[y] for x,y in zip(df['TEXT'], df['LOCATION'])]

train['ABV'] = createFeature(train)
valid['ABV'] = createFeature(valid)
# test['ABV'] = createFeature(test)

grouped = train.groupby(by=['ABV', 'LABEL'], as_index = False, sort = False).count()
grouped = grouped.sort_values(by='TEXT', ascending = False)

topAbv = grouped['ABV'][:100]

train = train[train['ABV'].isin(topAbv)]
valid = valid[valid['ABV'].isin(topAbv)]
# test = test[test['ABV'].isin(topAbv)]

# Lets create a function to remove all the Punctuations from Text
def removePunctuation(df):
    return [t.translate(str.maketrans('','',string.punctuation)) for t in df['TEXT']]

# Lets create a function to Tokenize the Text column of dataset
def createTokens(df):
    return df['TEXT'].apply(lambda x: x.split(' '))

#Lets create a function to drop "Abstract_id", "Location" and "TEXT" columns from dataset
def dropCols(df):
    return df.drop(columns=['ABSTRACT_ID', 'LOCATION', 'TEXT'])

# Lets create a function to remove stop words from the Text column
def removeStop(df):
    stopWords = spacy.lang.en.stop_words.STOP_WORDS
    # Remove any stopwords which appear to be an Abbreviation
    [stopWords.remove(t) for t in df['ABV'].str.lower() if t in stopWords]
    return df['TOKEN'].apply(lambda x: [item for item in x if not item in stopWords])

def tolower(df):
    return [t.lower() for t in df['TEXT']]

def preProcessData(df):   
    df['TEXT'] = tolower(df)
    df['TEXT'] = removePunctuation(df)
    df['TOKEN'] = createTokens(df)
    df = dropCols(df)
    df['TOKEN'] = removeStop(df)
    return df

# Lets load the train dataset.
train = preProcessData(train)
valid = preProcessData(valid)
# test = preProcessData(test)


abbrev = list(train['ABV'].unique())
valid = valid[valid['ABV'].isin(abbrev)]
# test = test[test['ABV'].isin(abbrev)]
labels = list(train['LABEL'].unique())
valid = valid[valid['LABEL'].isin(labels)]
# test = test[test['LABEL'].isin(labels)]

# Lets tag every Token List with its Label

train_tagged = train.apply(lambda x: TaggedDocument(words = x['TOKEN'], tags = [x['LABEL']]), axis=1)
valid_tagged = valid.apply(lambda x: TaggedDocument(words = x['TOKEN'], tags = [x['LABEL']]), axis=1)
# test_tagged = test.apply(lambda x: TaggedDocument(words = x['TOKEN'], tags = [x['LABEL']]), axis=1)

# # Convert TOKEN column from string to list
# train['TOKEN'] = train['TOKEN'].apply(lambda x: ast.literal_eval(x))
# valid['TOKEN'] = valid['TOKEN'].apply(lambda x: ast.literal_eval(x))
# test['TOKEN'] = test['TOKEN'].apply(lambda x: ast.literal_eval(x))


# from sklearn.preprocessing import LabelEncoder
# Instantiate the LabelEncoder
label_encoder = LabelEncoder()

def prepare_text_and_labels(dataset, include_labels=True):
    """
    Prepares text variables and labels for abbreviation expansion tasks.
    
    Args:
        dataset (pd.DataFrame): The dataset containing 'TOKEN' and optionally 'LABEL' columns.
        include_labels (bool): Whether to return the labels along with the texts.
        
    Returns:
        list: A list of input texts.
        list (optional): A list of labels if include_labels is True.
    """
    # Generate input texts from the 'TOKEN' column
    texts = [" ".join(tokens) for tokens in dataset["TOKEN"]]
    
    # If labels are needed, extract them
    if include_labels and "LABEL" in dataset.columns:
        labels = dataset["LABEL"].tolist()
        return texts, labels
    
    return texts

# Assuming `train`, `valid`, and `test` are pandas DataFrames from your notebook

# For training set
train_texts, train_labels = prepare_text_and_labels(train)

# For validation set
valid_texts, valid_labels = prepare_text_and_labels(valid)

# For test set, if you don't need labels
# test_texts = prepare_text_and_labels(test, include_labels=False)

# Display examples
print(train_texts[:2])  # Example texts
print(train_labels[:2])  # Corresponding labels

# Fit the LabelEncoder on the training labels
label_encoder.fit(train_labels)

# Transform the training labels (if needed)
train_labels_encoded = label_encoder.transform(train_labels)

# Filter validation set to remove unseen labels
valid_filtered = valid[valid["LABEL"].isin(label_encoder.classes_)]
valid_texts, valid_labels = prepare_text_and_labels(valid_filtered)

# Transform the validation labels
valid_labels_encoded = label_encoder.transform(valid_labels)




# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# import torch
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
# import numpy as np

def train_and_evaluate_abbreviation_model(train_texts, train_labels, valid_filtered, model_name="bert-base-uncased", output_dir="./trained_model", num_epochs=3, batch_size=16, learning_rate=2e-5):
    """
    Trains a BERT-based model for abbreviation expansion and evaluates on a filtered validation dataset.
    
    Args:
        train_texts (list): List of training texts.
        train_labels (list): List of training labels.
        valid_filtered (pd.DataFrame): Filtered validation dataset with 'TOKEN' and 'LABEL' columns.
        model_name (str): Pretrained model name from Hugging Face.
        output_dir (str): Directory to save the trained model.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        dict: Evaluation metrics including precision, recall, F1-score, and accuracy.
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize training data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    
    # Encode labels for training
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    
    # Prepare validation texts and labels
    valid_texts, valid_labels = prepare_text_and_labels(valid_filtered)
    
    # Tokenize validation data
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=128)
    valid_labels_encoded = label_encoder.transform(valid_labels)
    
    # Create PyTorch datasets
    class AbbreviationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = AbbreviationDataset(train_encodings, train_labels_encoded)
    valid_dataset = AbbreviationDataset(valid_encodings, valid_labels_encoded)
    
    # Load pretrained model
    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model on the validation set
    valid_preds = trainer.predict(valid_dataset)
    
    # Extract true and predicted labels
    predicted_labels = torch.argmax(torch.tensor(valid_preds.predictions), axis=1).numpy()
    true_labels = valid_labels_encoded
    
    # Compute evaluation metrics
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_)
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
    
    # Return metrics and other results
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }
    
    return metrics



metrics = train_and_evaluate_abbreviation_model(
    train_texts=train_texts,
    train_labels=train_labels,
    valid_filtered=valid_filtered,  # Validation data filtered for unseen labels
    model_name="bert-base-uncased",
    output_dir="./trained_model",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
)

# Display metrics
print("Evaluation Metrics:")
for key, value in metrics.items():
    if key == "confusion_matrix":
        print(f"{key}:\n{value}")
    else:
        print(f"{key}: {value}")
