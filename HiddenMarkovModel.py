from nltk import FreqDist, ConditionalFreqDist


class HiddenMarkovModel:

    def __init__(self, sentences):
        self.sentences = sentences

    def calculate_transition_prob(self):

        # remove the inner lists from the sentences
        # Get all the tags in the sentences and
        # Calculate the frequency distribution of every tag in the tag set

        unified_sentences = [tup for sent in self.sentences for tup in sent]
        tags_in_sentences = [tag for (_, tag) in unified_sentences]
        tags_frequency_distributions = FreqDist(tags_in_sentences)

        # Calculate the conditional Frequency Distribution of tag2 given that tag1 appears before t2 C(t2,t1)
        # Get all distinct POS/TAG (types) and the total number of types.
        tag_conditional_freq_distribution = ConditionalFreqDist((sentence[i - 1][1], sentence[i][1]) for sentence in \
                                                                self.sentences for i in range(1, len(sentence)))
        tag_type_set = tag_conditional_freq_distribution.conditions()
        tag_types_number = len(tag_type_set)

        # TODO
        # Create a matrix. Use + 1 to add the <s> Start symbol. # MUST ADD + 1 FOR <S>!!!!!!!!!!1
        transition_pob_matrix = [[0 for i in range(tag_types_number)] for j in range(tag_types_number)]

        # Iterate through the tag set for each row
        # Fill each row with the transition probability P(t2|t1)/P(t1) by iterating again through the tag set.
        for outer_counter, row_tag in enumerate(tag_type_set):
            collocations_with_current_tag = list(tag_conditional_freq_distribution.items())[outer_counter]

            for inner_counter, column_tag in enumerate(tag_type_set):
                row_column_tag_collocations = collocations_with_current_tag[1][column_tag]
                row_tag_frequency = tags_frequency_distributions[row_tag]
                transition_pob_matrix[outer_counter][inner_counter] = row_column_tag_collocations / row_tag_frequency
