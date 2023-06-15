""" This script is written to extract noun phrases that are likely to refer to the meaning of HUMAN BEING, PERSON.
This part is crucial in improving the captions generated by the auto-captions model
"""


import stanza
import os
from stanza.pipeline.core import DownloadMethod 
from nltk.corpus import wordnet as wn
from typing import Union

HOME = os.getcwd()

# this is the library that will perform most of the analysis on text
NLP = stanza.Pipeline('en', download_method=DownloadMethod.REUSE_RESOURCES, use_gpu=True) # to avoid downloading the models every time

# this is the synonyms' set of the word 'person' that best serves our purposes
PEOPLE = wn.synset('people.n.01')
# this is the synonyms' set of the word 'people' that best serves our purposes
PERSON = wn.synset('person.n.01')

def words_tags(tree:stanza.models.constituency.parse_tree.Tree) -> list[tuple[str, str]]:
    """ This function extracts the tokens and their POS tags in the textual chunk represented 
    by the root of the passed consitituency tree

    Args:
        tree (stanza.models.constituency.parse_tree.Tree): _description_

    Returns:
        list[tuple[str, str]]: a list of tokens and and their corresponding POS tags at the leafs of a given root of constituency tree
    """
    return [(x.label, x.children[0].label) for x in tree.yield_preterminals()]



def person_word(word:str, pos: str, threshold: float=0.2, synset_reference=wn.synset) -> bool:
    """This function determines whether a given word could possibly denote a 'HUMAN BEING', 'PERSON'

    Args:
        word (str): the given word
        pos (str): the POS tag associated with the word in the text
        threshold (float, optional): the minimal similarity between the given word and the reference word in the wordnet meanings' tree. Defaults to 0.2.
        synset_reference a set of synonyms from word net. Defaults to None.

    Returns:
        bool: _description_
    """

    if synset_reference is None:
        synset_reference = PERSON # this way we can use PEOPLE as well with the same function

    if pos.startswith('v'):
        pos = 'VERB'
    
    is_person = False

    try:
        l = wn.synsets(word, pos=pos.lower()[0])
        for meaning in l[:3]: # make sure to include the pos tag in the problem
            # unfrequent use cases of words
            if meaning.path_similarity(PERSON) >= threshold:
                is_person = True
                break
    except KeyError as e:
        # word's lemma might not belong to wordnet
        # in this case the output is False
        pass

    return is_person




def num_person_words(noun_phrase: Union[list[str], list[tuple[str, str]]], meta_data: dict, known_person_words: set) -> tuple[list[str], set]:
    """This function given a piece of text, (either with or without the POS tags) extract the words that could be interpreted as person 

    Args:
        noun_phrase (list[tuple[str, str]]): The piece of text (generally representing a noun phrase)
        meta_data (dict): The lemmas and POS tags extracted by the stanza library for the text in question
        known_person_words (set): words that were already detected as PERSON: added as a performance boost

    Returns:
        tuple[list[str], set]: a tuple of the words that could mean PERSON in this text , a set of words known as PERSON in general (for further iterations) 
    """


    # the meta data contains both the lemms and POS tags
    pos_tags = [meta_data[t[1].lower()][0] for t in noun_phrase]
    lemmas = [meta_data[t[1].lower()][1] for t in noun_phrase]
    
    person_lists = [(index, l) for index, (l, pos) in enumerate(zip(lemmas, pos_tags)) if pos == 'NOUN' and 
                    (l in known_person_words or person_word(l, pos, synset_reference=PERSON) or person_word(l, pos, synset_reference=PEOPLE))]    

    # make sure to add the words that could possibly mean PERSON to the set
    known_person_words.update([t[1] for t in person_lists])
    
    return person_lists, known_person_words


def get_NP_components(root:stanza.models.constituency.parse_tree.Tree, 
                      meta_data: dict, person_np: set=None):
    """Given a piece of text represented by the root of its constituency tree, extract all the noun phrases
    that contain 'PERSON' words (with additional constraints)

    Args:
        root (stanza.models.constituency.parse_tree.Tree): the root of the constituency tree of a given piece of text
        meta_data (dict): lemmas and POS tags of the tokens in the given text
        person_np (set, optional): set of previously seen PERSON words. Defaults to None.

    Returns:
        _type_: _description_
    """
    # this function is supposed to return the largest NPs including exactly one word with a close meaning to 'PERSON'  
    if person_np is None:
        person_np = set()
    
    result = []
    # only consider Noun Pthrases
    if root.label == 'NP':
        # if the current root reprsents a Noun Phrase, 
        noun_phrase = words_tags(root)
        # extract the number of words that represent a PERSON
        num_persons, person_np = num_person_words(noun_phrase, meta_data, person_np)

        if len(num_persons) == 1:
            # the word that could mean person can be preceeded only with Adjectives (ADJ) or determinents
            # first extract the position of such word
            pos_word = num_persons[0][0]
            good = False
            for i in range(pos_word):
                good = meta_data[noun_phrase[i][1].lower()][0] not in ['NOUN', 'VERB']
                if not good:
                    for child in root.children:
                        result.extend(get_NP_components(child, meta_data, person_np))
            if good:
                result.append(noun_phrase)

        elif len(num_persons) >= 2:
            # one exception for this condition is having all the words (with approximately PERSON meaning ) on a row
            if [n[0] for n in num_persons] == list(range(num_persons[0][0], num_persons[0][0] + len(num_persons))): 
                result.append(noun_phrase)
            
            else:                
                for child in root.children:
                    result.extend(get_NP_components(child, meta_data, person_np))

    else:
        for child in root.children:
            result.extend(get_NP_components(child, meta_data, person_np))

    return result 



def convert_to_text(pos_words: list[tuple[str, str]], filter:bool) -> str:
    """_summary_

    Args:
        l (list[tuple[str, str]]): a list of tuples (POS_tag, text)
        filter (bool): whether to filter the filler words: mainly non-adjectives and non-nouns

    Returns:
        str: the list converted to text
    """
    try:
        if filter:
            return " ".join([t[1] for t in pos_words if t[0] in ['NN', 'JJ']]).strip().lower() 
        return " ".join([t[1] for t in pos_words]).strip().lower()
    except:
        print("Error with converting. Pos words: \n", pos_words)
        return pos_words

def extract_NP_text(text: Union[str,list], nlp_object=None, plain_text:bool=True, filter:bool=True) -> list[list[str]]:
    """This function given a list of text, will extract all the possible PERON noun phrases in the each text
    return a list of the same length where each element corresponds to the different NPs in the corresponding sentence

    Args:
        text (str): either a list of sentences, or the sentences concatenated all together in one big text
        nlp_object (_type_, optional): the stanza model responsible for processing the text
        plain_text (bool, optional): whether the components should be returned as plain text or tuples of (POS, text). Defaults to True.
        filter (bool, optional): whether the text should be filtered from stop / filler words. Defaults to True.

    Returns:
        _type_: _description_
    """
    # set the defeault NLP object
    if nlp_object is None:
        nlp_object = NLP

    # initialize objects to save the results
    np_components = []
    person_words = set()

    # if isinstance(text, str):
        # create a doc for the text
        
    doc = nlp_object(text)
    print(len(doc.sentences))
    # iterate through sentences
    for s in doc.sentences:
        # the constituency tree
        tree = s.constituency
        # the root generally contains redundant information for our purposes
        c = tree.children[0]
        meta_data = dict([(w.text.lower(), [w.upos, w.lemma]) for w in s.words]) # the assumption is as follows: if the word is repeated then it is frequent and the lemma and POS tag is the same
        np_components.append(get_NP_components(c, meta_data, person_words))

    # elif isinstance(text, list):
    #     for sentence in text:
    #         doc = nlp_object(sentence)
    #         for s in doc.sentences:
    #             # the constituency tree
    #             tree = s.constituency
    #             # the root generally contains redundant information for our purposes
    #             c = tree.children[0]
    #             meta_data = dict([(w.text.lower(), [w.upos, w.lemma]) for w in s.words]) # the assumption is as follows: if the word is repeated then it is frequent and the lemma and POS tag is the same
    #             np_components.append(get_NP_components(c, meta_data, person_words))


    # apply the needed post-processing
    if plain_text:
        return [[convert_to_text(t, filter=filter) for t in component] for component in np_components]
    
    return np_components
