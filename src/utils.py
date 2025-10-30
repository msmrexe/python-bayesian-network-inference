import logging
import os
from itertools import product

# --- Logging Setup ---

def setup_logging():
    """Configures logging to file and console."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("networkx").setLevel(logging.WARNING)

# --- Factor Class for Variable Elimination ---

class Factor:
    """
    Represents a factor in the variable elimination algorithm.
    A factor is defined over a set of variables and stores a probability
    for each combination of their values.
    """
    def __init__(self, variables, cpt):
        """
        Initialize a Factor.
        
        Args:
            variables (list): A list of variable names (e.g., ['A', 'B']).
            cpt (dict): A dictionary mapping assignments (tuples) to probabilities.
                        e.g., {(0, 0): 0.1, (0, 1): 0.9, ...}
        """
        self.variables = list(variables)
        self.cpt = cpt

    def __str__(self):
        return f"Factor({self.variables}) \n{self.cpt}"

    def get_assignment_prob(self, assignment):
        """Gets prob for a full assignment dict."""
        try:
            key = tuple(assignment[var] for var in self.variables)
            return self.cpt[key]
        except KeyError:
            return 0.0

def multiply_factors(factor1, factor2):
    """Multiplies two factors."""
    
    # 1. Find all variables
    all_vars = sorted(list(set(factor1.variables) | set(factor2.variables)))
    new_cpt = {}
    
    # 2. Iterate over all possible assignments for the combined variables
    # Assuming binary variables (0 or 1) for this project context
    for assignment_values in product([0, 1], repeat=len(all_vars)):
        assignment = dict(zip(all_vars, assignment_values))
        
        # 3. Get probability from each factor
        prob1 = factor1.get_assignment_prob(assignment)
        prob2 = factor2.get_assignment_prob(assignment)
        
        # 4. Store the product
        new_cpt[assignment_values] = prob1 * prob2
        
    return Factor(all_vars, new_cpt)

def sum_out(factor, var_to_sum_out):
    """Sums out a variable from a factor."""
    
    # 1. Find remaining variables
    new_vars = [v for v in factor.variables if v != var_to_sum_out]
    var_index = factor.variables.index(var_to_sum_out)
    new_cpt = {}
    
    # 2. Iterate over all assignments of remaining variables
    for assignment_values in product([0, 1], repeat=len(new_vars)):
        new_key = assignment_values
        summed_prob = 0.0
        
        # 3. Sum over the values (0 and 1) of the variable being removed
        for var_value in [0, 1]:
            old_assignment_list = list(assignment_values)
            old_assignment_list.insert(var_index, var_value)
            old_key = tuple(old_assignment_list)
            
            if old_key in factor.cpt:
                summed_prob += factor.cpt[old_key]
                
        new_cpt[new_key] = summed_prob
        
    return Factor(new_vars, new_cpt)

def reduce_factor(factor, evidence_var, evidence_value):
    """Reduces a factor based on observed evidence."""
    
    # 1. Find new variables and index of the evidence variable
    new_vars = [v for v in factor.variables if v != evidence_var]
    var_index = factor.variables.index(evidence_var)
    new_cpt = {}

    # 2. Iterate over all assignments of the *original* variables
    for old_key, prob in factor.cpt.items():
        # 3. If the assignment matches the evidence...
        if old_key[var_index] == evidence_value:
            # 4. ...add it to the new CPT with the evidence variable removed
            new_key = tuple(v for i, v in enumerate(old_key) if i != var_index)
            new_cpt[new_key] = prob
            
    return Factor(new_vars, new_cpt)

def normalize_factor(factor):
    """Normalizes a factor so its probabilities sum to 1."""
    total = sum(factor.cpt.values())
    if total == 0:
        return factor
    
    normalized_cpt = {key: val / total for key, val in factor.cpt.items()}
    return Factor(factor.variables, normalized_cpt)
