import  numpy as np
class Viterbi:

    def __init__(self, HMM_model):

        self.HMM_model = HMM_model
        self.transition_pob_matrix = self.HMM_model.get_transition_prob_for_tags()
        self.emission_pob_matrix = self.HMM_model.get_emission_prob()
        self.tag_types_number = self.HMM_model.get_num_of_tag_types()

        self.tag_type_set = self.HMM_model.get_tag_type_set()

    def tag_words(self, sentence):

        # Create the viterbi prop matrix
        # Create the back pointer list
        # Set previous path prob to 1 at first because it always start from the starting state
        viterbi_probs = [[0 for i in range(self.tag_types_number)] for j in range(len(sentence))]

        tag_type_set = self.HMM_model.get_tag_type_set()
        words_set = self.HMM_model.get_word_type_set()

        for tag_counter, tag in enumerate(tag_type_set):
            emission_prob = self.emission_pob_matrix[tag].prob(sentence[0])
            viterbi_probs[0][tag_counter] = self.transition_pob_matrix['<s>'].prob(tag) * emission_prob

        for word_counter, word in enumerate(sentence[1:]):
            for tag_counter, tag in enumerate(tag_type_set):

                emission_prob = self.emission_pob_matrix[tag].prob(word)

                all_transition_prob = [self.transition_pob_matrix[tag_type_set[i]].prob(tag) for i in range (self.tag_types_number)]
                viterbi_probs[word_counter + 1][tag_counter] = max(np.multiply(viterbi_probs[word_counter], all_transition_prob) * emission_prob)

        return  self.decoding_viterbi(sentence,viterbi_probs)


    def decoding_viterbi(self,sentence,viterbi_probs):

        tags = []
        #Calculate the prob of the tag given </s>
        final_prob = [self.transition_pob_matrix[self.tag_type_set[i]].prob('</s>') for i in range(self.tag_types_number)]
        print(max(np.multiply(viterbi_probs[len(sentence) - 1], final_prob)))
        tag_index = np.argmax(np.multiply(viterbi_probs[len(sentence)-1], final_prob))
        tag = self.tag_type_set[tag_index]
        tags.insert(0,tag)

        for i in range(len(sentence) - 1, 0, -1):
            final_prob = [self.transition_pob_matrix[self.tag_type_set[j]].prob(tag) for j in range(self.tag_types_number)]
            tag_index = np.argmax(np.multiply(viterbi_probs[i - 1], final_prob) * self.emission_pob_matrix[tag].prob(sentence[i]))
            tag = self.tag_type_set[tag_index]
            tags.insert(0,tag)
        print(len(tags))
        return tags
