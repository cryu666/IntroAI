import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        Return two parameters including:
            model: the number of co-occurrence of each pair
            features: the pair of each pattern
        E.g.,
            model: (I -> am): 10, (I -> want): 5, ...
        '''
        # TO-DO 1-1: (You can use the hint code between begin and end or just delete it.)
        # begin your code
        
        # model = {}
        
        # sentence = list(np.concatenate(corpus_tokenize).flat)
        
        # review_bigram_frequency = nltk.ngrams(sentence, self.n)
        # review_bigram_model = nltk.ngrams(sentence, self.n)


        # for pair in review_bigram_model:
    
        #     first_word, second_word = pair
        #     if second_word == "[CLS]":
        #         continue
        #     if first_word not in model:
        #         model[first_word] = {}
        #     if second_word not in model[first_word]:
        #         model[first_word][second_word] = 1
        #     else:
        #         model[first_word][second_word] += 1
        
        # features = Counter(review_bigram_frequency)      #given

        # return model, features


        model = defaultdict(lambda: defaultdict(lambda: 0))   # given
        features = Counter()                   # given
        
        # Compute co-occurrence of each pair of words
        for sentence in corpus_tokenize:
          for i in range(len(sentence)-1):
            model[sentence[i]][sentence[i+1]] += 1
            features[(sentence[i], sentence[i+1])] += 1
        
        return model, features
        
        # end your code
        
        
        
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        perplexity = 0
        for document_tokenize in corpus:
            twograms = nltk.ngrams(document_tokenize, self.n)
            N = len(list(nltk.ngrams(document_tokenize, self.n)))
            probabilities = []
            for w1, w2 in twograms:
                numerator = 1 + self.model[w1][w2]
                denominator = sum(self.model[w1].values())
                # give a value to avoid divide-by-zero
                if denominator == 0:
                    probabilities.append(1e-3)
                else:
                    probabilities.append(numerator / denominator)

            cross_entropy = -1 / N * sum([math.log(p, 2) for p in probabilities])
            perplexity += math.pow(2, cross_entropy)
        
        perplexity /= len(corpus)
        return perplexity
    
    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # TO-DO 1-2
        # begin your code

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 500
        
        features_keys = sorted(self.features, key=self.features.get, reverse=True)

        features_list = features_keys[:feature_num]

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        
        train_corpus_embedding = []
        test_corpus_embedding = []
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        for sentence in train_corpus:
            train_bigrams = [(sentence[i], sentence[i+1]) for i in range(len(sentence)-1)]
            vector = np.zeros(len(features_list))
            for bigram in train_bigrams:
                if bigram in features_list:
                    index = features_list.index(bigram)
                    vector[index] += 1

            train_corpus_embedding.append(vector)
        
        for sentence in test_corpus:
            test_bigrams = [(sentence[i], sentence[i+1]) for i in range(len(sentence)-1)]
            vector = np.zeros(len(features_list))
            for bigram in test_bigrams:
                if bigram in features_list:
                    index = features_list.index(bigram)
                    vector[index] += 1

            test_corpus_embedding.append(vector)


        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
