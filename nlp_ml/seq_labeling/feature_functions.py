from nlp_ml.seq_labeling.POSEnglishTag import POSEnglishTag


class FeatureExtractions:
    # typical adverb
    def adverb_featrue(self, word_list, word_idx, current_label, prev_label):
        if current_label == POSEnglishTag.ADV \
                and word_list[word_idx].endswith("ly"):
            return 1
        return 0

    # question sentence
    def admire_feature(self, word_list, word_idx, current_label, prev_label):
        if current_label == POSEnglishTag.VERB and \
                        word_idx == 1 and \
                word_list[-1].endswith("?"):
            return 1
        return 0

    # adjective noun pattern
    def adj_noun_feature(self, word_list, word_idx, current_label,
                         prev_label):
        if prev_label == POSEnglishTag.ADJ and \
                        current_label == POSEnglishTag.NOUN:
            return 1
        return 0

    # continuous preposition (inappropriate pattern)
    def continuig_prep_featruen(self, word_list, word_idx, current_label,
                                prev_label):
        if prev_label == POSEnglishTag.PRT and \
                        current_label == POSEnglishTag.PRT:
            return 0
        return 1
