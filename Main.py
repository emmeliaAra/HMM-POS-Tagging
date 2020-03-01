from nltk.corpus import brown
from HiddenMarkovModel import HiddenMarkovModel
from Viterbi import Viterbi

# Divide testing and training corpus
trainSetSize = 10000
testingSetSize = 500

sentences = brown.tagged_sents(tagset='universal')
trainSet = sentences[0:trainSetSize]
testSet = sentences[trainSetSize:trainSetSize + testingSetSize]  # Continue from where the training set stopped.

# create an instance of the HHM and passed the training set to generate its parameters.
hiddenMarkovModel = HiddenMarkovModel(trainSet)
hiddenMarkovModel.calculate_transition_prob()
hiddenMarkovModel.calculate_emission_prob()


unified_test_set = [tup for sent in testSet for tup in sent]
test_set_observations = [w for (w, _) in unified_test_set]
for test in testSet[:1]:

    test_set_observations = [w for (w, _) in test]
    viterbi = Viterbi(hiddenMarkovModel, test_set_observations)
