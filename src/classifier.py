import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


def read_file(path):
    with open(path, "r") as f:
        lines = [line for line in f if line.strip()]
        splitted_lines = [line.strip().split("\t") for line in lines]
        labels, categories, target, position, sentences = zip(*splitted_lines)

    df = pd.DataFrame(
        {
            "sentence": sentences,
            "category": categories,
            "target": target,
            "position": position,
            "label": labels,
        }
    )

    df[["start", "end"]] = df["position"].str.split(":", expand=True)
    df.drop("position", axis=1, inplace=True)

    return df


def transform(df):
    return (
        df.reset_index()
        .rename(columns={"index": "sentence_id", "sentence": "s1", "category": "s2"})[
            ["sentence_id", "s1", "s2", "label"]
        ]
        .to_dict("records")
    )


def to_dataloader(path, label_map, model_name):
    if path:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        df = pd.DataFrame(transform(read_file(path)))
        labels = list(map(lambda l: label_map[l], df.label))

        encoded = tokenizer(
            list(df.s1),
            list(df.s2),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        return torch.utils.data.TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long).reshape(-1, 1),
        )

    else:
        return None


def get_data(model_name, trainfile=None, devfile=None, datafile=None):
    label_map = {"neutral": 0, "negative": 1, "positive": 2}

    datasets = {
        "train": to_dataloader(trainfile, label_map, model_name),
        "dev": to_dataloader(devfile, label_map, model_name),
        "test": to_dataloader(datafile, label_map, model_name),
    }

    return datasets


def train_model(config: dict, train_set, val_set):
    best_model = 0
    best_accuracy = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = config["model_id"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    num_labels = 3

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    for epoch in range(config["epochs"]):
        print("Training (Epoch " + str(epoch + 1) + ")")

        for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            loss = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )[0]

            loss.backward()
            optimizer.step()

        val_true = []
        val_pred = []
        for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
            with torch.no_grad():
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )[1]

                val_true += labels
                val_pred += list(torch.softmax(logits, -1).argmax(-1).cpu())

        acc = accuracy_score(val_true, val_pred)
        print("Validation Accuracy: ", acc)

        if acc > best_accuracy:
            best_model = model
            best_accuracy = acc
            print("Best accuracy: ", acc, ", Epoch", (epoch + 1))

    print("Training complete")

    return best_model


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.config = {
            "model_id": "distilbert-base-uncased",
            "model_name": "distilbert",
            "epochs": 5,
            "batch_size": 25,
            "lr": 1e-5,
            "labels": ["neutral", "negative", "positive"],
        }

    #############################################
    def train(self, trainfile, devfile="../data/devdata.csv"):
        """Trains the classifier model on the training set stored in file trainfile"""

        config = self.config

        train = read_file(trainfile)

        data = get_data(config["model_id"], trainfile=trainfile, devfile=devfile)

        train_processed = data["train"]
        dev_processed = data["dev"]

        self.model = train_model(
            config, train_set=train_processed, val_set=dev_processed
        )

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        config = self.config
        final_labels = ["neutral", "negative", "positive"]

        model = self.model

        data = get_data(config["model_id"], datafile=datafile)

        ds = data["test"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_loader = torch.utils.data.DataLoader(
            ds, batch_size=int(config["batch_size"]), shuffle=False
        )

        model.eval()

        pred = []

        for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
            with torch.no_grad():
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )[1]

                pred += list(torch.softmax(logits, -1).argmax(-1).cpu())

        pred = [final_labels[tens.item()] for tens in pred]

        return pred
