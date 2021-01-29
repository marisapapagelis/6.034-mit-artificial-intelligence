# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = set()
    for parent in net.get_parents(var):
        ancestors.add(parent)
        ancestors = ancestors.union(get_ancestors(net, parent))
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = set()
    for parent in net.get_children(var):
        descendants.add(parent)
        descendants = descendants.union(get_descendants(net, parent))
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    nondescendants = set()
    descendants = get_descendants(net, var)
    for variable in net.get_variables():
        if variable not in descendants:
            nondescendants.add(variable)
    nondescendants.remove(var)
    return nondescendants


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    simplify = dict()
    parents = net.get_parents(var)
    descendants = get_descendants(net, var)
    given = set(givens.keys())
    if parents.issubset(given):
        for p in given.difference(parents):
            if p in descendants:
                return givens
        for p in given:
            if p in parents:
                simplify.setdefault(p,givens[p])
        return simplify
    return givens
    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    if givens is None:
        try:
            prob = net.get_probability(hypothesis)
            return prob
        except ValueError:
            raise LookupError
    given = simplify_givens(net, list(hypothesis)[0], givens)
    try:
        prob = net.get_probability(hypothesis, given)
        return prob
    except ValueError:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    vars = net.topological_sort()
    vars.reverse()
    cond = hypothesis.copy()
    prob = 1.0

    for var in vars:
        value = cond.pop(var)
        if cond == dict():
            t = probability_lookup(net, {var: value}, None)
        else:
            t = probability_lookup(net, {var: value}, cond)
        prob *= t
    return prob
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    joint_prob = net.combinations(net.get_variables(), hypothesis)
    prob = 0
    for joint in joint_prob:
        prob += probability_joint(net, joint)
    return prob

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens is None:
        return probability_marginal(net, hypothesis)
    for hyp in hypothesis:
        if hyp in givens:
            if hypothesis[hyp] != givens[hyp]:
                return 0
            
    num = probability_marginal(net, dict(hypothesis, **givens))
    denom = probability_marginal(net, givens)
    prob_cond = num/denom

    return prob_cond
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net, hypothesis, givens)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    num = 0
    vars = net.get_variables()

    for var in vars:
        domain = len(net.get_domain(var)) - 1
        parents = net.get_parents(var)
        
        if len(parents) == 0:
            num += domain
        else:
            b = 1
            for p in parents:
                b *= len(net.get_domain(p))
            num += domain * b
    return num


#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    combinations = net.combinations([var1, var2])
    
    for comb in combinations:
        if givens is None:
            prob1 = probability(net, {var1: comb[var1]}, None)
            prob2 = probability(net, {var1: comb[var1]}, {var2: comb[var2]})

        else:
            prob1 = probability(net, {var1: comb[var1]}, givens)
            prob2 = probability(net, {var1: comb[var1]}, dict(givens, **{var2: comb[var2]}))

        if not(approx_equal(prob1, prob2)):
            return False
    return True
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    ancestors = get_ancestors(net, var1).union(get_ancestors(net, var2))
    if givens is not None:
        for given in givens:
            ancestors = get_ancestors(net, given).union(ancestors)
        givensList = list(givens.keys())
    else:
        givensList = list()
    
    ancestorList = list(ancestors)
    newNet = net.subnet(ancestors.union(set([var1, var2] + givensList)))

    for ancestor in ancestors:
        children1 = newNet.get_children(ancestor)
        ancestorList.remove(ancestor)
        for anc in ancestorList:
            children2 = newNet.get_children(anc)
            if len(children1.intersection(children2)) != 0:
                newNet.link(ancestor, anc)

    newNet = newNet.make_bidirectional()
    if givens is not None:
        for given in givens:
            newNet.remove_variable(given)
        
    final_path = newNet.find_path(var1, var2)

    return True if final_path is None else False


#### SURVEY ####################################################################

NAME = 'Marisa Papagelis'
COLLABORATORS = 'Peyton Wang'
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = 'Independence'
WHAT_I_FOUND_BORING = 'Probability'
SUGGESTIONS = None
