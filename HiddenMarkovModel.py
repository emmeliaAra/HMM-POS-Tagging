import nltk
from nltk import FreqDist, WittenBellProbDist


class HiddenMarkovModel:

    def __init__(self, sentences):
        self.sentences = sentences
        self.unified_sentences = {}

        self.tags_in_sentences = {}
        self.tag_type_set = {}
        self.tag_types_number = 0

        self.words_in_sentences = {}
        self.word_types_set = {}
        self.word_types_number = 0

        self.emission_prob = {}
        self.transition_prob_for_tag = {}

        self.set_up_variables()

    def set_up_variables(self):
        self.unified_sentences = [tup for sent in self.sentences for tup in sent]
        self.tags_in_sentences = [tag for (_, tag) in self.unified_sentences]
        self.words_in_sentences = [word for (word, _) in self.unified_sentences]

        self.tag_type_set = sorted(set(self.tags_in_sentences)) + ['<s>'] + ['</s>']
        print(self.tag_type_set)
        self.tag_types_number = len(self.tag_type_set)
        self.word_types_set = sorted(set(self.words_in_sentences))
        self.word_types_number = len(self.word_types_set)

    def calculate_transition_prob_for_POS_tags(self):

        tags_in_sentence = self.reformat_sentences()
        tag_bigrams = nltk.bigrams(tags_in_sentence)

        for tag in self.tag_type_set:
            tag_bigrams = nltk.bigrams(tags_in_sentence)
            words = [t2 for (t1, t2) in tag_bigrams if t1 == tag]
            self.transition_prob_for_tag[tag] = WittenBellProbDist(FreqDist(words), bins=1e6)

    def reformat_sentences(self):
        new_tag_in_sent = []
        for sent in self.sentences:
            tags_in_sentence = [t for (w, t) in sent]
            new_tag_in_sent = new_tag_in_sent + ['<s>'] + tags_in_sentence + ['</s>']
        return new_tag_in_sent

    def calculate_emission_prob(self):

        for tag in self.tag_type_set:
            words = [w for (w, t) in self.unified_sentences if t == tag]
            self.emission_prob[tag] = WittenBellProbDist(FreqDist(words), bins=1e6)

    def get_num_of_observations(self):
        return self.word_types_number

    def get_num_of_tag_types(self):
        return self.tag_types_number

    def get_tag_type_set(self):
        return self.tag_type_set

    def get_word_type_set(self):
        return self.word_types_set

    def get_emission_prob(self):
        return self.emission_prob

    def get_transition_prob_for_tags(self):
        return self.transition_prob_for_tag
