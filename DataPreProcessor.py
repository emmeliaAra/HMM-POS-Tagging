from collections import Counter

import re


class DataPreProcessor:

    def __init__(self, train_data_set, test_data_set):
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set

        self.unified_sentences = [tup for sent in self.train_data_set for tup in sent]
        self.unified_test_set = [tup for sent in self.test_data_set for tup in sent]

        self.infrequent_words = []
        self.infrequent_words_testing = []

        # self.identify_infrequent_words()
        # self.tag_capital_words()
        # self.tag_UNI_ing_words()
        # # self.tag_numbers()
        #
        # self.identify_infrequent_words_in_testing_corpus()
        # self.tag_capital_words_testing()
        # self.tag_UNI_ing_words_testing()
        # # self.tag_numbers_testing()

    def identify_infrequent_words(self):
        # Find words occurring only once in data set

        words = [w for (w, _) in self.unified_sentences]
        words_count = Counter(words)
        self.infrequent_words = [w for (w, c) in words_count.items() if c == 1]
        return self.infrequent_words

    def identify_infrequent_words_in_testing_corpus(self):

        words_in_train = [w for (w, _) in self.unified_sentences]
        train_word_set = sorted(set(words_in_train))

        words_in_test = [w for (w, _) in self.unified_test_set]
        test_word_set = sorted(set(words_in_test))

        self.infrequent_words_testing = [w for w in test_word_set if
                                         train_word_set.count(w) == 0 or self.infrequent_words.count(w) > 0]
        return self.infrequent_words_testing

    def tag_capital_words(self,infrequent_words,data_set):
        new_data_set = []
        for sent in data_set:
            new_data_set.append(
                [("UNK", t) if w[0].isupper() and w != sent[0][0] and infrequent_words.count(w)>0 else (w, t) for
                 (w, t) in sent])
        return new_data_set

    def tag_UNI_ing_words(self,infrequent_words,data_set):
        str1 = " "
        new_data_set = []
        ing_list = re.findall(r'\b(\w+ing)\b', str1.join(infrequent_words))
        for sent in data_set:
            new_data_set.append(
                [("UNK-ing", t) if ing_list.count(w) > 0 else (w, t) for (w, t) in sent])
        return new_data_set

    def tag_numbers(self,infrequent_words,data_set):
        str1 = " "
        new_data_set = []
        number_list = re.findall(r'\d+[,.-]\d+ | [$%£]\d+|\b\d+\b', str1.join(infrequent_words))

        for sent in data_set:
            new_data_set.append([("NUMBER", t) if number_list.count(w) > 0 else (w, t) for (w, t) in sent])
        return new_data_set