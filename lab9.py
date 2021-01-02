# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    result = {}
    weight = make_fraction(1, len(training_points))
    for point in training_points:
        result[point] = weight
    return result

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    result = {}
    for classifier, points in classifier_to_misclassified.items():
        weights = 0
        for point in points:
            weights += point_to_weight[point]
        result[classifier] = weights
    return result

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    classifier_sorted = sorted(classifier_to_error_rate)
    best_error = None
    best_class = None
    if use_smallest_error is True:
        f = lambda x: x
        g = lambda x: best_error > x
    else:
        f = lambda x: abs(make_fraction(1,2) - x)
        g = lambda x: best_error < x
    for classifier in classifier_sorted:
        value = f(make_fraction(classifier_to_error_rate[classifier]))
        if best_error is None or g(value):
            best_error = value
            best_class= classifier
    if make_fraction(classifier_to_error_rate[best_class]) == make_fraction(1,2):
        raise NoGoodClassifiersError("'best' weak classifier has an error rate of exactly 1/2")
    return best_class

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 1.0:
        return -INF
    elif error_rate == 0.0:
        return INF
    result = make_fraction(1,2) * ln(make_fraction(1-error_rate, error_rate))
    return result

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassified = set()
    for point in training_points:
        score = 0
        for classifier, voting_power in H:
            if point not in classifier_to_misclassified[classifier]:
                score += voting_power
            else:
                score -= voting_power
        if score <= 0:
            misclassified.add(point)
    return misclassified

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    misclassified_points = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    if len(misclassified_points) > mistake_tolerance:
        return False
    return True

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for point, old_weight in point_to_weight.items():
        if point not in misclassified_points:
            new_weight = make_fraction(1,2) * make_fraction(1, 1 - error_rate) * old_weight
        else:
            new_weight = make_fraction(1,2) * make_fraction(1, error_rate) * old_weight
        point_to_weight[point] = new_weight

    return point_to_weight


#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    H = list()
    round = 0
    point_to_weight = initialize_weights(training_points)
    while round < max_rounds:
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
        except:
            return H
        error_rate = classifier_to_error_rate[best_classifier]
        voting_power = calculate_voting_power(error_rate)
        H.append((best_classifier, voting_power))

        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H

        point_to_weight = update_weights(point_to_weight, classifier_to_misclassified[best_classifier], error_rate)
        round += 1
    return H


#### SURVEY ####################################################################

NAME = 'Marisa Papagelis'
COLLABORATORS = 'Peyton Wang'
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = 'adaboost'
WHAT_I_FOUND_BORING = 'calculate error rates'
SUGGESTIONS = None
