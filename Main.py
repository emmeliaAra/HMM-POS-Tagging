from nltk.corpus import brown
from HiddenMarkovModel import HiddenMarkovModel

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
