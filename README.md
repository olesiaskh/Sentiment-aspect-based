# Sentiment classifier (BERT-based)

The problem is treated as a sentence-pair classification task. The model takes as input a sentence pair (a sentence and a category name) and outputs a label (polarity). 
The full process consists of loading and pre-processing the dataset, training the model, and predicting labels.

## Pre-processing
Filter dataset to include only sentence-category-label columns and encode labels ('neutral': 0, 'negative': 1, 'positive': 2). Apply AutoTokenizer (padding=True, truncation=True, max_length=512) to sentence-category pairs, save ids and attention masks. Finally, convert the pre-processed dataset to a tensor.

## Training 
Fine-tune distilbert-base-uncased using the following hyperparameters: (epochs: 5, batch_size: 25, lr: 1e-5), and AdamW() optimizer. Save the best model based on accuracy score, and use it to predict labels (sentiment) in the final step.

Note: the code is adapted by Olesia Khrapunova from an original project from CentraleSupelec written by Suiyan Liu, Alain Mikael Alafriz, Olesia Khrapunova and Milind Bhatnagar
