# MIT 6.034 Lab 3: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    return list() in csp.domains.values()

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for constraint in csp.get_all_constraints():
        val1 = csp.get_assignment(constraint.var1)
        val2 = csp.get_assignment(constraint.var2)
        
        if not (val1 is None or val2 is None): 
            if not constraint.check(val1, val2):
                return False
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    
    agenda = [problem]
    solution = None
    num_ext = 0

    while len(agenda) > 0:
        prob = agenda.pop(0)  
        num_ext += 1
        if has_empty_domains(prob) or not check_all_constraints(prob):
            continue
        
        if not prob.unassigned_vars:
            solution = prob.assignments
            return (solution, num_ext)
       
        var = prob.pop_next_unassigned_var()
       
        extensions = []
        for val in prob.get_domain(var):
            new_prob = prob.copy()
            new_prob.set_assignment(var, val)
            extensions.append(new_prob)

        agenda = extensions + agenda

    return (solution, num_ext)


# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    domain = dict()
    copy = csp.copy()
    neighbors = csp.get_neighbors(var)
    for neighbor in neighbors:
        constraints = csp.constraints_between(neighbor, var)
        if len(constraints) > 1:
            csp.set_domain(neighbor,list())
            return None
        else:
            for val1 in csp.get_domain(neighbor):
                count = 0
                for val2 in csp.get_domain(var):
                    if not constraints[0].check(val1,val2):
                        count += 1
                if count == len(csp.get_domain(var)):
                    copy.eliminate(neighbor,val1)
                    csp.set_domain(neighbor,copy.get_domain(neighbor))
                    if len(csp.get_domain(neighbor)) == 0:
                        return None
                    domain.setdefault(neighbor)
    return sorted(domain.keys())

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    if has_empty_domains(problem):
        return None, 1
    count = 0
    agenda = [problem]
    while len(agenda) > 0:
        popped = agenda.pop(0)
        count += 1
        if not has_empty_domains(popped) and check_all_constraints(popped):
            unassigned = popped.pop_next_unassigned_var()
            if unassigned is None:
                return (popped.assignments,count)
            l = list()
            for p in popped.get_domain(unassigned):
                copy = popped.copy().set_assignment(unassigned,p)
                forward_check(copy,unassigned)
                l.append(copy)
            l.extend(agenda)
            agenda = l
    return (None,count)


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    result = []
    if queue is None:
        values = csp.get_all_variables()
    else:
        values = queue
    while values:
        value = values.pop(0)
        result.append(value)
        elim = eliminate_from_neighbors(csp,value)
        if elim:
            for neighbor in elim:
                if neighbor not in values:
                    values.append(neighbor)
        if elim is None:
            return None
        
    return result


# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    if has_empty_domains(problem):
        return (None, 1)
    count = 0
    agenda = [problem]
    while len(agenda) > 0:
        popped = agenda.pop(0)
        count += 1
        if not has_empty_domains(popped) and check_all_constraints(popped):
            unassigned = popped.pop_next_unassigned_var()
            if unassigned is None:
                return (popped.assignments,count)
            l = list()
            for p in popped.get_domain(unassigned):
                copy = popped.copy().set_assignment(unassigned,p)
                domain_reduction(copy,[unassigned])
                l.append(copy)
            l.extend(agenda)
            agenda = l
    return (None,count)


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue is None:
        all_variables = csp.get_all_variables()[:]
        queue = all_variables
    result = list()
    while queue:
        variable = queue.pop(0)
        result.append(variable)
        for v in csp.variables:
            for c in csp.constraints_between(variable, v):
                dom = csp.get_domain(v)[:]
                for d in dom:
                    len_var = len(csp.get_domain(variable))
                    count = 0
                    for value in csp.get_domain(variable):
                        if not c.check(value, d):
                            count += 1
                    if count is len_var:
                        csp.eliminate(v, d)
                        if len(csp.get_domain(v)) is 0:
                            return None
                        if not v in queue:
                            if enqueue_condition_fn(csp, v):
                                queue.append(v)
    return result

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return len(csp.get_domain(var)) == 1

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    if has_empty_domains(problem):
        return (None, 1)
    count = 0
    agenda = [problem]
    while len(agenda) > 0:
        problem = agenda.pop(0)
        count += 1
        if not has_empty_domains(problem) and check_all_constraints(problem):
            unassigned = problem.pop_next_unassigned_var()
            if unassigned is None:
                return (problem.assignments,count)
            c = list()
            for v in problem.get_domain(unassigned):
                csp = problem.copy().set_assignment(unassigned,v)
                if enqueue_condition != None:
                    propagate(enqueue_condition, csp,[unassigned])
                c.append(csp)
            c.extend(agenda)
            agenda = c
    return (None,count)

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = 8


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return n in [m-1, m+1]

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not constraint_adjacent(m, n)

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    result = list()
    for i in range(len(variables)-1):
        for j in range(len(variables)-i-1):
            result.append(Constraint(variables[i],variables[i+j+1],constraint_different))
    return result


#### SURVEY ####################################################################

NAME = 'Marisa Papagelis'
COLLABORATORS = 'Peyton Wang'
HOW_MANY_HOURS_THIS_LAB_TOOK = 15
WHAT_I_FOUND_INTERESTING = 'custom constraints'
WHAT_I_FOUND_BORING = 'generic constraints'
SUGGESTIONS = 'nothing'
