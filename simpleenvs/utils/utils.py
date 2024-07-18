from collections import defaultdict
from typing import List, Tuple, Hashable
from numbers import Number


def reduce_prob_tuples(tuple_list: List[Tuple[Hashable, Number]]) -> List[Tuple[Hashable, Number]]:
    """
    Takes a list of tuples of the form (element, probability), and merges any duplicate elements by summing their probabilities.

    Args:
        tuple_list (List[Tuple[Hashable, Number]]): A list of tuples of the form (element, probability).

    Returns:
        List[Tuple[Hashable, Number]]: A list of tuples of the form (element, probability) with all duplicates merged.
    """
    prob_dict = defaultdict(float)
    for element, prob in tuple_list:
        prob_dict[element] += prob

    return list(prob_dict.items())
