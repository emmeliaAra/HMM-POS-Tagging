from nltk.corpus import brown
from HiddenMarkovModel import HiddenMarkovModel
from Viterbi import Viterbi

import time

start = time.time()

# Divide testing and training corpus
trainSetSize = 10000
testingSetSize = 500

sentences = brown.tagged_sents(tagset='universal')
trainSet = sentences[0:trainSetSize]
testSet = sentences[trainSetSize:trainSetSize + testingSetSize]  # Continue from where the training set stopped.

# create an instance of the HHM and passed the training set to generate its parameters.
hiddenMarkovModel = HiddenMarkovModel(trainSet)
hiddenMarkovModel.calculate_transition_prob_for_POS_tags()
hiddenMarkovModel.calculate_emission_prob()

unified_test_set = [tup for sent in testSet for tup in sent]
test_set_tags = [t for (_, t) in unified_test_set]
viterbi = Viterbi(hiddenMarkovModel)
viterbi_tags = []
for test in testSet:

    #print(len(test), " ...........................")
    test_observations = [w for (w, _) in test]
    viterbi_tags += viterbi.tag_words(test_observations)


check = [v_tag for v_tag, t_tag in zip(viterbi_tags, test_set_tags) if v_tag == t_tag]
viterbi_accuracy = len(check)/len(test_set_tags)

print("Percentage", viterbi_accuracy * 100)
print(test_set_tags)
print(viterbi_tags)
