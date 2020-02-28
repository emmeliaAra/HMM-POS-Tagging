import numpy


class HiddenMarkovModel:

    def __init__(self, sentences):
        self.sentences = sentences

    def calculate_transition_prob(self):
        # remove the inner lists from the sentences
        # Get all the tags in the sentences
        # Get all distinct types of tags
        # Get the number of all tags

        unified_sentences = [tup for sent in self.sentences for tup in sent]
        tags_in_sentences = [tag for (_, tag) in unified_sentences]  # or do it like this  [pair[1] for pair in unified_sentences]
        tag_type_set = set(tags_in_sentences)
        tag_types_numbers = len(tag_type_set)

        # Create a matrix. Use + 1 to add the <s> Start symbol.
        transition_pob_matrix = numpy.zeros((tag_types_numbers + 1, tag_types_numbers), dtype='float32')

        
