from nltk import FreqDist, ConditionalFreqDist, WittenBellProbDist


class HiddenMarkovModel:

    def __init__(self, sentences):
        self.sentences = sentences

    def calculate_transition_prob(self):

        # Remove the inner lists from the sentences Get all the tags in the sentences and Calculate the frequency
        # Distribution of every POS tag in the tag set Sets the frequency distribution for the <s> tag
        #       -> equal with the total number of sentences because all sentences have a starting point
        unified_sentences = [tup for sent in self.sentences for tup in sent]
        tags_in_sentences = [tag for (_, tag) in unified_sentences]
        tags_frequency_distributions = FreqDist(tags_in_sentences)
        starting_tag_frequency = len(self.sentences)

        # Calculate the conditional Frequency Distribution of tag2 given that tag1 appears before t2 C(t2,t1)
        tag_conditional_freq_distribution = self.calculate_transition_prob_for_POS_tags()
        starting_tag_conditional_freq_distribution = self.calculate_transition_prob_for_start_tag()

        # Get all distinct POS/TAG (types) and the total number of types.
        # Create a matrix. Use + 1 to add the <s> Start symbol.
        tag_type_set = tag_conditional_freq_distribution.conditions()
        tag_types_number = len(tag_type_set)
        transition_pob_matrix = [[0 for i in range(tag_types_number)] for j in range(tag_types_number + 1)]
        print(len(transition_pob_matrix))

        # Iterate through the tag set for each row
        # Fill each row with the transition probability P(t2|t1)/P(t1) by iterating again through the tag set.
        for outer_counter, row_tag in enumerate(tag_type_set):
            collocations_with_current_tag = list(tag_conditional_freq_distribution.items())[outer_counter]

            for inner_counter, column_tag in enumerate(tag_type_set):
                row_column_tag_collocations = collocations_with_current_tag[1][column_tag]
                row_tag_frequency = tags_frequency_distributions[row_tag]
                transition_pob_matrix[outer_counter + 1][
                    inner_counter] = row_column_tag_collocations / row_tag_frequency

        # Iterate through the cells for the first row (<s>) to update the collocations with <s>.
        for counter, column_tag in enumerate(tag_type_set):
            collocations_with_start_tag = list(starting_tag_conditional_freq_distribution.items())[0]
            start_column_tag_collocation = collocations_with_start_tag[1][column_tag]
            transition_pob_matrix[0][counter] = start_column_tag_collocation / starting_tag_frequency

    def calculate_transition_prob_for_start_tag(self):
        sentences = self.reformat_sentences()
        return ConditionalFreqDist((sentence[1], sentence[0]) for sentence in sentences)

    def calculate_transition_prob_for_POS_tags(self):

        # Calculate the conditional Frequency Distribution of tag2 given that tag1 appears before t2 C(t2,t1)
        # Get all distinct POS/TAG (types) and the total number of types.
        return ConditionalFreqDist((sentence[i - 1][1], sentence[i][1]) for sentence in \
                                   self.sentences for i in range(1, len(sentence)))

    def reformat_sentences(self):

        temp_sent = list()
        for sent in self.sentences:
            temp = sent
            temp_sent.append([temp[0][1], '<s>'])

        return temp_sent

    def calculate_emission_prob(self):

        unified_sentences = [tup for sent in self.sentences for tup in sent]
        tags_in_sentences = [tag for (_, tag) in unified_sentences]
        words_in_sentences = [word for (word, _) in unified_sentences]

        tag_types_set = set(tags_in_sentences)
        word_types_set = set(words_in_sentences)

        tag_types_number = len(tag_types_set)
        word_types_number = len(word_types_set)

        emission_pob_matrix = [[0 for i in range(word_types_number)] for j in range(tag_types_number)]
        smoothed = {}
        for tag in tag_types_set:
            words = [w.lower() for (w, t) in unified_sentences if t == tag]
            smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        for outer_counter, tag in enumerate(tag_types_set):
            for inner_counter, word in enumerate(word_types_set):
                emission_pob_matrix[outer_counter][inner_counter] = smoothed[tag].prob(word.lower())