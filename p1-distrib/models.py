# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import math
from collections import defaultdict


def sigmoid(x: float) -> float:
    """
    Compute the sigmoid function: 1 / (1 + e^-x)
    :param x: input value
    :return: sigmoid output between 0 and 1
    """
    return 1 / (1 + math.exp(-x))

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        return
    
    def extract_features(self, sentence, add_to_indexer = False):
        features = Counter()
        for word in sentence:
            word = word.lower()
            word = word.strip('.,!?')
            features[word] = 1
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
            return
    
    def extract_features(self, sentence, add_to_indexer=False):
        features = Counter()

        cleaned = []
        for word in sentence:
            word = word.lower().strip('.,!?')
            if word:
                cleaned.append(word)
                features[word] = 1 

        for i in range(len(cleaned) - 1):
            bigram = cleaned[i] + "_" + cleaned[i + 1]
            features[bigram] = 1

        return features
    
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self,weights,feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence):
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        score = 0.0
        for f, count in features.items():
            if f in self.weights:
                score += self.weights[f] * count

        prob = sigmoid(score)
        return 1 if prob >= 0.5 else 0

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs = 100, lr=.001) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    weights = defaultdict(float)
    for iteration in range(num_epochs):
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
            # compute score
            score = sum(weights[f] * count for f, count in features.items())
            prob = sigmoid(score)
            # update weights
            error = ex.label - prob
            for f, count in features.items():
                weights[f] += lr * error * count
    classifier = LogisticRegressionClassifier(weights,feat_extractor)
    return classifier



def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor,args.num_epochs,args.lr)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, network: nn.Module, word_embeddings: WordEmbeddings):
        self.network = network
        self.word_embeddings = word_embeddings

    def predict(self, words: List[str]) -> int:
        indices = [
            self.word_embeddings.word_indexer.index_of(w)
            if self.word_embeddings.word_indexer.contains(w)
            else self.word_embeddings.word_indexer.index_of("UNK")
            for w in words
        ]
        
        indices_tensor = torch.tensor([indices], dtype=torch.long)

        self.network.eval()
        with torch.no_grad():
            logits = self.network(indices_tensor)

        return int(torch.argmax(logits).item())

    def predict_all(self, all_words: List[List[str]]) -> List[int]:
        return [self.predict(words) for words in all_words]



class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings, hidden_size=200, num_labels=2):
        super().__init__()

        # Embedding layer initialized from pretrained vectors
        self.embedding = word_embeddings.get_initialized_embedding_layer()

        embed_dim = self.embedding.embedding_dim

        self.g = nn.ReLU()
        self.embeddingToHidden = nn.Linear(embed_dim, hidden_size)
        self.hiddenToOutput = nn.Linear(hidden_size, num_labels)
        self.log_softmax = nn.LogSoftmax(dim=0)
    
    def forward(self, word_indices):
       
        embeds = self.embedding(word_indices) 
        avg_embed = torch.mean(embeds, dim=1) 
        
        h = self.g(self.embeddingToHidden(avg_embed)) 
        
        logits = self.hiddenToOutput(h) 
        
        return logits



def make_batch(exs, word_embeddings):
    max_len = max(len(ex.words) for ex in exs)
    batch_indices = []

    for ex in exs:
        idxs = [
            word_embeddings.word_indexer.index_of(w)
            if word_embeddings.word_indexer.contains(w)
            else word_embeddings.word_indexer.index_of("UNK")
            for w in ex.words
        ]
        idxs += [0] * (max_len - len(idxs)) 
        batch_indices.append(idxs)

    return torch.tensor(batch_indices, dtype=torch.long)



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    model = DeepAveragingNetwork(word_embeddings)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32

    for epoch in range(args.num_epochs):
        random.shuffle(train_exs)

        for i in range(0, len(train_exs), batch_size):
            batch = train_exs[i:i+batch_size]
            optimizer.zero_grad()

            inputs = make_batch(batch, word_embeddings)
            labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
    return NeuralSentimentClassifier(model, word_embeddings)
