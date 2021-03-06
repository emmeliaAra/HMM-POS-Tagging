from nltk.corpus import brown
from HiddenMarkovModel import HiddenMarkovModel
from Viterbi import Viterbi
from DataPreProcessor import DataPreProcessor
import matplotlib.pyplot as plt

import time

start = time.time()

# Divide testing and training corpus
trainSetSize = 10000
testingSetSize = 500

sentences = brown.tagged_sents(tagset='universal')
trainSet = sentences[0:trainSetSize]
testSet = sentences[trainSetSize:trainSetSize + testingSetSize]  # Continue from where the training set stopped.

dataPreProcessor = DataPreProcessor(trainSet,testSet)

training_infrequent_words = dataPreProcessor.identify_infrequent_words()
trainSet = dataPreProcessor.tag_capital_words(training_infrequent_words, trainSet)
trainSet = dataPreProcessor.tag_UNI_ing_words(training_infrequent_words, trainSet)
trainSet = dataPreProcessor.tag_numbers(training_infrequent_words, trainSet)

testing_infrequent_words = dataPreProcessor.identify_infrequent_words_in_testing_corpus()
testSet = dataPreProcessor.tag_capital_words(testing_infrequent_words, testSet)
testSet = dataPreProcessor.tag_UNI_ing_words(testing_infrequent_words, testSet)
testSet = dataPreProcessor.tag_numbers(testing_infrequent_words, testSet)

# create an instance of the HHM and passed the training set to generate its parameters.
hiddenMarkovModel = HiddenMarkovModel(testSet)
hiddenMarkovModel.calculate_transition_prob_for_POS_tags()
hiddenMarkovModel.calculate_emission_prob()

unified_test_set = [tup for sent in testSet for tup in sent]
test_set_tags = [t for (_, t) in unified_test_set]

viterbi = Viterbi(hiddenMarkovModel)
viterbi_tags = []
for test in testSet:
    if len(test)<100:
        test_observations = [w for (w, _) in test]
        viterbi_tags += viterbi.tag_words(test_observations)

check = [v_tag for v_tag, t_tag in zip(viterbi_tags, test_set_tags) if v_tag == t_tag]
viterbi_accuracy = len(check)/len(test_set_tags)

print("Initial tags", len(test_set_tags))
print("Correct tags", len(check))
print("Percentage", viterbi_accuracy * 100)
print(test_set_tags)
print(viterbi_tags)