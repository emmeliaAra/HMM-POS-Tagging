
class Viterbi:

    def __init__(self, HMM_model, observations):

        self.HMM_model = HMM_model
        self.transition_pob_matrix = self.HMM_model.get_transition_prob_matrix()
        self.emission_pob_matrix = self.HMM_model.get_emission_prob_matrix()
        self.tag_types_number = self.HMM_model.get_num_of_tag_types()
        self.word_types_number = self.HMM_model.get_num_of_observations()

        self.tag_words(observations)

    def tag_words(self, sentence):

        # Create the viterbi prop matrix
        # Create the back pointer list
        # Set previous path prob to 1 at first because it always start from the starting state
        viterbi_probs = [[0 for i in range(self.word_types_number)] for j in range(self.tag_types_number)]
        back_pointer = {}
        previous_path_prob = 1
        previouse_state = "<s>"
        previouse_state_index = 0

        words_set = self.HMM_model.get_word_type_set()

        for word_counter, word in enumerate(sentence[:1]):
            for tag_counter, tag in enumerate(self.HMM_model.get_tag_type_set()):
                if words_set.count(word) > 0:
                    word_index = words_set.index(word)
                    transition_prob = self.transition_pob_matrix[previouse_state_index][tag_counter]
                    emission_prob = self.emission_pob_matrix[tag_counter][word_index]

                    viterbi_probs[tag_counter][word_counter] = previous_path_prob * transition_prob * emission_prob

            values_for_word = [column[word_counter] for column in viterbi_probs]
            max_prop = max(values_for_word)
            print(max_prop)
            ##find index of state and set the values p leis kato.. i think after this to mono p emine en na dis ti enna kamis me ta unkown words... 

            # if words_set.count(word) > 0:
            #     viterbi_probs[]

            # HERE TAKE THE MAX VALUE FROM VITERBIMATRIX(THN SIGKEKRIMENI COLUMN) AND MAKE
            # THE PREVI STATE THE CURRENT STATE
            # THE PREBIOUSE PATH PROB  TO THE MAX VALUE IN THAT ROW.
