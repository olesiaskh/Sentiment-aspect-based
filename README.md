# Aspect-based sentiment analysis

The problem is treated as a sentence pair classification task. The model takes as input a sentence pair (a sentence and a category name) and outputs a label (polarity). 
The full process consists of loading and pre-processing the dataset, training the model, and predicting labels.

## Pre-processing
Filter dataset to include only sentence-category-label columns, encode labels ('neutral': 0, 'negative': 1, 'positive': 2). Apply AutoTokenizer to sentence-category pairs, save ids and attention masks. Finally, convert the pre-processed dataset to a tensor.

## Training 
Fine-tune a BERT model using different hyper parameters and save the best one (based on accuracy score).

Note: the code is adapted by Olesia Khrapunova from a project from NLP course (MS in Data Science and Business Analytics, CentraleSupelec). The original code was written by Suiyan Liu, Alain Mikael Alafriz, Milind Bhatnagar and Olesia Khrapunova.
