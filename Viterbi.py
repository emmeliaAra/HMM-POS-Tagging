class Viterbi:

    def __init__(self, HMM_model):

        self.HMM_model = HMM_model
        self.transition_pob_matrix = self.HMM_model.get_transition_prob_matrix()
        self.emission_pob_matrix = self.HMM_model.get_emission_prob_matrix()
        self.tag_types_number = self.HMM_model.get_num_of_tag_types()
        self.word_types_number = self.HMM_model.get_num_of_observations()

    def tag_words(self, sentence):

        # Create the viterbi prop matrix
        # Create the back pointer list
        # Set previous path prob to 1 at first because it always start from the starting state
        viterbi_probs = [[0 for i in range(self.word_types_number)] for j in range(self.tag_types_number)]
        back_pointer = []
        previous_path_prob = 1
        previous_state = "<s>"
        previous_state_index = 0
        tag_type_set = self.HMM_model.get_tag_type_set()

        words_set = self.HMM_model.get_word_type_set()

        for word_counter, word in enumerate(sentence):
            for tag_counter, tag in enumerate(tag_type_set):
                if words_set.count(word) > 0: # if the word not in the word_set then add the smoothing prob. #TODO fix.
                    word_index = words_set.index(word)
                    emission_prob = self.emission_pob_matrix[tag_counter][word_index]
                else:
                    emission_prob = 1e7

                transition_prob = self.transition_pob_matrix[previous_state_index + 1][tag_counter]
                viterbi_probs[tag_counter][word_counter] = previous_path_prob * transition_prob * emission_prob

            values_for_word = [column[word_counter] for column in viterbi_probs]
            max_prop = max(values_for_word)
            current_state_index = values_for_word.index(max_prop)

            previous_path_prob = max_prop
            previous_state = list(tag_type_set)[current_state_index]
            previous_state_index = current_state_index
            back_pointer.append(previous_state)

        return back_pointer
