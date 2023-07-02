from collections import Counter
import warnings
import numpy as np
import pandas as pd

from .noun_phrases_detection import extract_noun_phrases


def build_captions_class_matrix(filtered_nps: list[list[list[str]]], predictions: list[list[str]]) \
        -> tuple[dict[str, Counter], dict[str, set]]:
    """This function computes the needed statistics to associate a score with
     each pair of noun phrase and predicted label

    Args:
        filtered_nps (list[list[str]]): the i-th element represents a list of noun phrases extracted from the i-th text
        predictions (list[list[str]]): the i-th elements represents a list of face labels
        predicted on the i-th image (associated with the i-th text)

    Returns:
        tuple[Counter, Counter]: class_np_counter: the class frequency of each word in the given noun phrases, 
        np_class_counter: the set of classes seen with each word in the given noun phrases 
    """

    # to save the frequency of each NP with each class
    prediction_np_counter = {}

    # to save which classes each NP was seen with: used mainly for the inverse document frequency component
    np_prediction_counter = {}

    for np_list, pred_list in zip(filtered_nps, predictions):
        # the first step is to find the terms / tokens associated with every label
        for p in pred_list:
            if p not in prediction_np_counter:
                prediction_np_counter[p] = Counter()  # a dictionary for each of the classes

            # iterate through the noun phrases: 
            for np_tokens in np_list:
                # increase the frequency of each term in the caption, in the dictionary associated with the class
                prediction_np_counter[p].update(dict([(token, 1) for token in np_tokens]))

                for token in np_tokens:
                    if token not in np_prediction_counter:
                        np_prediction_counter[token] = set()
                        # add the pred to the list of classes 'word' is associated with
                    np_prediction_counter[token].add(p)

    return prediction_np_counter, np_prediction_counter


def find_decided_captions(filtered_noun_phrases: list[list[list[str]]], predictions: list[list[str]]) \
        -> dict[str, Counter]:
    """This function executes the same functionality as the previous function with the main difference of considering
    only pairs of (predictions, noun phrases) for which the number of both is equal to one.
    Such occurrences should weight more in the final score of a pair:
    (np, label)
    Args:
        filtered_noun_phrases (list[list[str]]):
        the i-th element represents a list of noun phrases extracted from the i-th text
        predictions (list[list[str]]):
        the i-th elements represents a list of face labels predicted on the i-th image (associated with the i-th text)

    Returns:
        tuple[Counter, Counter]: class_np_counter: the class frequency of each word in the given noun phrases, 
        np_prediction_info: the set of classes seen with each word in the given noun phrases
    """

    # this function will just return the captions and face_predictions with length 1
    decisive_list = [(noun_phrases[0], p[0]) for noun_phrases, p in zip(filtered_noun_phrases, predictions) if
                     len(noun_phrases) == len(p) == 1]
    
    if len(decisive_list) == 0:
        # return an empty dictionary
        return {}

    # convert the list of tuples to 2 lists
    # nps represents a list of strings 
    # predictions: a list of strings
    noun_phrases, predictions = list(map(list, zip(*decisive_list)))

    # build a counter to map each class to its decided captions
    decisive_class_np_counter = {}

    for np_tokens, pred in zip(noun_phrases, predictions):

        if pred not in decisive_class_np_counter:
            decisive_class_np_counter[pred] = Counter()

        decisive_class_np_counter[pred].update(dict([(w, 1) for w in np_tokens]))

    return decisive_class_np_counter


def np_prediction_score(noun_phrases: list[str],
                        prediction: str,
                        prediction_np_counter: dict[str, Counter],
                        np_prediction_counter: dict[str, set],
                        decided_prediction_np: dict[str, Counter]) -> float:
    """This function assigns a score of a give pair: noun_phrase, class_label.
    It is a customized version of tf-idf representation of a text with respect to a
    given vocabulary and set of documents

    Args:
        noun_phrases (str): a noun phrase that was previously seen in the entire set of noun phrases
        prediction (str): a label
        prediction_np_counter (Counter):
        np_prediction_counter (Counter):
        decided_prediction_np (Counter):

    Returns:
        float: the score of the pair 
    """
    # set the default values of prediction_np_counter to an empty counter
    
    def word_class_score(word: str):
        frequency_score = prediction_np_counter[prediction][word] if prediction in prediction_np_counter else 0
        decided_freq_score = decided_prediction_np[prediction][word] if prediction in decided_prediction_np else 0

        numerator = 1 + frequency_score + decided_freq_score

        # the denominator: the number of unique classes the word was associated with + 1 
        denominator = 1 + len(np_prediction_counter[word])

        return np.log(numerator / denominator) + 1

    return float(np.mean([word_class_score(w) for w in noun_phrases]))


def map_predictions(noun_phrases: list[list[str]],
                    predictions: list[str],
                    prediction_np_counter: dict[str, Counter],
                    np_prediction_counter: dict[str, set],
                    decided_prediction_np: dict[str, Counter]) -> dict[str: list[str]]:

    # since it is not possible to have lists as indices, we need to convert noun phrases (as lists) to strings
    # we need a mapping from the resultings strings to the original lists
    def to_str(tokens: list[str]) -> str:
        return (" ".join(tokens)).lower().strip()

    str_to_tokens = dict([(to_str(np_tokens), np_tokens) for np_tokens in noun_phrases])

    # create a dataframe to save the score of each noun phrase with the predicted class
    np_scores = pd.DataFrame(data=[], index=[to_str(n) for n in noun_phrases], columns=predictions, dtype=float)

    for noun_p in noun_phrases:
        for p in predictions:
            np_scores.at[to_str(noun_p), p] = \
                np_prediction_score(noun_p, p, prediction_np_counter, np_prediction_counter, decided_prediction_np)

    mapping = {}
    while not np_scores.empty:
        # first extract the highest score in the table
        max_score = np.amax(np_scores.values)

        # locate it
        indices, columns = np.where(np_scores == max_score)
        # extract the corresponding noun phrase and prediction
        best_np = list(np_scores.index)[indices[0]]
        best_pred = list(np_scores.columns)[columns[0]]

        # map the best prediction to the best 'noun phrase' as 'str' which is mapped to the noun phrase as 'list'
        mapping[best_pred] = str_to_tokens[best_np]

        # remove the index and the face from np_scores
        np_scores.drop(columns=best_pred, index=best_np, inplace=True)

    return mapping


def generate_captions(captions: list[str], predictions: list[list[list[str]]]) -> list[str]:

    nps = extract_noun_phrases(captions, select=True)
    noun_phrases, filtered_noun_phrases = list(map(list, zip(*nps)))

    # now we have the captions and the predictions ready

    # time to build the matrix
    prediction_np_counter, np_prediction_counter = build_captions_class_matrix(filtered_noun_phrases, predictions)

    # find the decided captions
    decided_prediction_np = find_decided_captions(filtered_noun_phrases, predictions)

    # iterate through each of the predictions and captions
    final_captions = []
    for np_list_index, (np_list, pred_list) in enumerate(zip(filtered_noun_phrases, predictions)):
        # map the noun phrase to the suitable class
        mapping = map_predictions(np_list,
                                  pred_list,
                                  prediction_np_counter,
                                  np_prediction_counter,
                                  decided_prediction_np)

        new_caption = captions[np_list_index]

        # for each pair of noun phrase and prediction
        for pred, np_tokens in mapping.items():
            # within the different nps in the current noun phrases, find the position 'np'
            np_index = np_list.index(np_tokens)
            # determine the exact text to replace 
            text_to_replace = noun_phrases[np_list_index][np_index]
            # text_to_replace is currently a list of strings: must be joined
            text_to_replace = " ".join(text_to_replace).strip().lower()
            # replace it in the caption
            new_caption = new_caption.replace(text_to_replace, pred)

        final_captions.append(new_caption)

    return final_captions
