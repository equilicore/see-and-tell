""" This script is written to extract noun phrases that are likely to refer to the meaning of HUMAN BEING, PERSON.
This part is crucial in improving the captions generated by the auto-captions model
"""
import os
import re
import numpy as np
import stanza

from stanza.pipeline.core import DownloadMethod
from nltk.corpus import wordnet as wn
from _collections_abc import Sequence
from typing import Union

HOME = os.getcwd()

# this is the library that will perform most text-analysis tasks
NLP = stanza.Pipeline('en', download_method=DownloadMethod.REUSE_RESOURCES, use_gpu=True)
# this is the synonyms' set of the word 'person' that best serves our purposes
PEOPLE = wn.synset('people.n.01')
# this is the synonyms' set of the word 'people' that best serves our purposes
PERSON = wn.synset('person.n.01')

NP = 'NP'
NOUN = 'NOUN'
VERB = 'VERB'
ADJECTIVE = 'ADJ'


def extract_words(tree: stanza.models.constituency.parse_tree.Tree) -> list[str]:
    """
    Args:
        tree: A constituency tree that represents the relations between the different tokens
        of a given sentence
    Returns: a list of words in the tree
    """
    return [x.children[0].label.lower() for x in tree.yield_preterminals()]


def is_word_person(word: str, pos_tag: str,
                   similarity_threshold: float = 0.2,
                   furthest_meaning: int = 3,
                   reference=None) -> bool:
    """
    This function applies a heuristic approach to determine whether a given word could possibly
    mean 'PERSON' or 'HUMAN BEING'
    Args:
        word: the word in question
        pos_tag: the POS tag of the given word
        similarity_threshold: The minimum similarity score the given word should achieve to be considered
                a 'PERSON' word
        furthest_meaning: a limit to the number of possible meanings for one word
        reference: either PERSON or PEOPLE

    Returns:
    whether the word can mean 'PERSON' or 'PEOPLE'
    """
    if reference is None:
        reference = PERSON

    if pos_tag.startswith('v'):
        pos_tag = 'VERB'

    is_person = False

    try:
        # synset uses the first character to represent the POS tag
        possible_meanings = wn.synsets(word, pos=pos_tag.lower()[0])[:furthest_meaning]

        for meaning in possible_meanings:
            sim = meaning.path_similarity(reference)
            if sim > similarity_threshold:
                is_person = True
                break

    except KeyError:
        # word's lemma might not belong to wordnet
        # in this case the output is False
        pass

    return is_person


def extract_person_words(noun_phrase_text: list[str], metadata: dict[str, tuple[str, str]], known_person_lemmas: set,
                         return_known_words: bool = False) \
        -> Union[list[tuple[int, str]], tuple[list[tuple[int, str]], set]]:
    """
    Extract all the words with a meaning close to 'PERSON' as well as their indices within the initial text
    Args:
        noun_phrase_text: the initial text
        metadata: a dictionary that maps each string in the original text to its POS and lemma

        known_person_lemmas: determining whether a word has a meaning close to 'PERSON' might be
        computationally expensive (relatively). This set is used as some sort of memoization to
        boost performance

        return_known_words: whether to return the set of known words

    Returns: a list of 'PERSON' words and their indices
    """

    def is_person(word: str) -> bool:
        """This inner function ensures easier use of the local variables,
        easier understanding and a bug-free implementation"""
        # extract the word's metadata
        pos_tag, lemma = metadata[word]
        result = pos_tag == 'NOUN' and (lemma in known_person_lemmas
                                        or is_word_person(lemma, pos_tag, reference=PERSON) or
                                        is_word_person(lemma, pos_tag, reference=PEOPLE))

        # add the lemma to the know_person_words
        if result:
            known_person_lemmas.add(lemma)

        return result

    result = [(index, word) for (index, word) in enumerate(noun_phrase_text) if is_person(word)]

    if return_known_words:
        return result, known_person_lemmas

    return result


def tokens_to_text(tokens: list[str], metadata: dict[str, tuple[str, str]], select: bool = False) -> str:
    if select:
        return (" ".join([t for t in tokens if metadata[t] in [NOUN, ADJECTIVE]])).lower().strip()

    return (" ".join(tokens)).lower().strip()


def __extract_noun_phrases_tree(root: stanza.models.constituency.parse_tree.Tree,
                                metadata: dict[str, tuple[str, str]],
                                known_person_lemmas: set) -> list[list[str]]:
    # let's first set some constraints to filter noun phrases
    def one_person_valid_np(words: list[str], person_words: list[tuple[int, str]]) -> bool:
        assert len(person_words) == 1, "ONLY 1 PERSON WORD IS EXPECTED"
        person_word_index, _ = person_words[0]
        # The words preceding the PERSON word cannot be VERBS or other NOUNS
        # as substituting such noun phrases would significantly alter the text's meaning
        noun_filter = [metadata[w][0] not in [NOUN, VERB] for w in words[:person_word_index]]
        return all(noun_filter)

    def more_persons_valid_np(words: list[str], person_words: list[tuple[int, str]]) -> bool:
        assert len(person_words) >= 2, "MULTIPLE PERSON WORDS EXPECTED"
        # NOUN PHRASES WITH MULTIPLE PERSONS SHOULD BE IGNORED EXPECT FOR THOSE THAT
        # 1. ONLY THE LAST PERSON HAS A POS TAG OF 'NOUN'
        # 2. HAVE THE PERSONS WORDS ALIGNED

        last_person_index, _ = person_words[-1]
        # noun_filter = all([metadata[w][0] not in [NOUN, VERB] for w in words[:last_person_index]])
        alignment_filter = [p_index for p_index, _ in person_words] == list(
            range(person_words[0][0], person_words[0][0] + len(person_words)))

        return alignment_filter

    result = []

    if root.label == NP:
        # extract the words
        words = extract_words(root)
        # extract the words in the nouns that represent a 'PERSON'
        person_words = extract_person_words(words, metadata, known_person_lemmas)

        if len(person_words) == 1 and one_person_valid_np(words, person_words) or \
                (len(person_words) >= 2 and more_persons_valid_np(words, person_words)):
            result.append(words)

        else:
            for child in root.children:
                result.extend(__extract_noun_phrases_tree(child, metadata, known_person_lemmas))

    else:
        for child in root.children:
            result.extend(__extract_noun_phrases_tree(child, metadata, known_person_lemmas))

    return result


def __prepare_str(string: str) -> str:
    REGEX1 = r'[^a-zA-Z\d\s-]+'
    REGEX2 = r'[^a-zA-Z\d\s]+$'
    # remove any punctuation whatsoever
    string = re.sub(REGEX1, '', string)
    # add '.' to the end of the sentence
    string += '.'
    # remove '-' if it is at the end of a sentence
    res = re.sub(REGEX2, '.', string.lower().strip())
    return res


def __almost_equal(s1: str, s2: str) -> bool:
    t1, t2 = min(s1, s2, key=len), max(s1, s2, key=len)
    return t1 in t2 and np.abs(len(t1) - len(t2)) <= 3


def __indices_mapping(cleaned: Sequence[str],
                      doc_sentences: list[str],
                      debug: bool = True) -> tuple[list[int], list[int]]:
    cleaned_set, doc_set = set(), set()
    cp, dp = 0, 0  # stands for cleaned_pointer, doc_pointer
    while dp < len(doc_sentences) and cp < len(cleaned):
        if cleaned[cp] == doc_sentences[dp]:
            cleaned_set.add(cp)
            doc_set.add(dp)
            cp += 1
            dp += 1
        else:
            temp_sentence = ''
            while cp < len(cleaned) and temp_sentence.strip() != doc_sentences[dp].strip():
                temp_sentence += f" {cleaned[cp]}"
                cp += 1
            dp += 1

    cleaned_indices = sorted(list(cleaned_set))
    doc_indices = sorted(list(doc_set))

    if debug:
        for i1, i2 in zip(cleaned_indices, doc_indices):
            c = cleaned[i1]
            d = doc_sentences[i2]
            assert __almost_equal(c, d)

    return cleaned_indices, doc_indices


def extract_noun_phrases(sentences: Sequence[str],
                         select: bool = False) -> tuple[list[list[list[str]]], list[int]]:
    cleaned = [__prepare_str(c) for c in sentences]
    # first prepare the sentences
    text = ' '.join(cleaned)

    # the produced text will be passed to NLP for processing
    doc = NLP(text)
    doc_sentences = [s.text.lower().strip() for s in doc.sentences]

    # since the tokenization process is not guaranteed to return the extract number of sentences
    # as in the input, some mapping of indices should take place.
    cleaned_indices, doc_indices = __indices_mapping(cleaned, doc_sentences)

    # keep in mind that noun_phrases function below works only
    # with doc.sentence object
    doc_sentences = [doc.sentences[i] for i in doc_indices]

    # define the set of known words
    known_person_words = set()

    def noun_phrases(s):
        # build the constituency tree
        tree = s.constituency
        # the root generally contains redundant information for our purposes
        c = tree.children[0]

        # save the lemma and POS tag for each word in the sentence
        metadata = dict([(w.text.lower(), (w.upos, w.lemma)) for w in s.words])
        # if the same word is repeated in the sentence, it will be associated with the last pair
        # of POS and lemma. The assumption here is that the word will be associated with the same meaning
        # and thus the same properties.
        result = __extract_noun_phrases_tree(c, metadata, known_person_words)

        if select:
            filtered_result = [[t for t in tokens if metadata[t][0] in [NOUN, ADJECTIVE]] for tokens in result]
            return result, filtered_result

        return result

    # let's save the result of the noun phrases
    nps = [noun_phrases(s) for s in doc_sentences]
    return nps, cleaned_indices
