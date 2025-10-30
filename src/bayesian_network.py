import networkx as nx
import matplotlib.pyplot as plt
import logging
from itertools import product
from .utils import Factor, multiply_factors, sum_out, reduce_factor, normalize_factor

log = logging.getLogger(__name__)

class BayesianNetwork:
    """
    Implements a Bayesian Network from scratch, supporting
    construction, visualization, and exact inference.
    """
    
    def __init__(self):
        """Initialize the Bayesian network as a directed acyclic graph (DAG)"""
        self.graph = nx.DiGraph()
        self.cpts = {}
        self.nodes = []
        log.info("Initialized empty Bayesian Network.")

    def add_node(self, node):
        """Add a node to the network"""
        if node not in self.graph:
            self.graph.add_node(node)
            self.nodes.append(node)
            log.debug(f"Added node: {node}")
        else:
            log.warning(f"Node {node} already exists.")

    def add_edge(self, parent, child):
        """Add a directed edge from parent to child"""
        if parent not in self.graph:
            log.error(f"Parent node {parent} does not exist. Add it first.")
            return
        if child not in self.graph:
            log.error(f"Child node {child} does not exist. Add it first.")
            return
            
        self.graph.add_edge(parent, child)
        log.debug(f"Added edge: {parent} -> {child}")
        
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent, child)
            log.error(f"Adding edge {parent} -> {child} creates a cycle. Edge removed.")
            raise ValueError("Cycles are not allowed in a Bayesian Network (DAG).")

    def set_cpt(self, node, cpt):
        """
        Set the conditional probability table for a node.
        
        CPT format:
        - For root node: {0: P(node=0), 1: P(node=1)}
        - For child node with parents (P1, P2):
          {(P1_val, P2_val): {0: P(node=0|...), 1: P(node=1|...)}, ...}
        """
        if node not in self.graph:
            log.error(f"Cannot set CPT for non-existent node: {node}")
            return
        
        self.cpts[node] = cpt
        log.debug(f"Set CPT for node: {node}")

    def draw_network(self, save_path=None):
        """Draw the Bayesian network using networkx"""
        plt.figure(figsize=(10, 7))
        try:
            # Use a layout that respects the hierarchy
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except ImportError:
            log.warning("pygraphviz not found. Using spring_layout. "
                        "For a hierarchical layout, run: pip install pygraphviz")
            pos = nx.spring_layout(self.graph)
            
        nx.draw(self.graph, pos, with_labels=True, node_size=4000, 
                node_color='#a0cbe2', font_size=12, font_weight='bold',
                arrowsize=20)
        plt.title("Bayesian Network Structure")
        
        if save_path:
            plt.savefig(save_path)
            log.info(f"Network graph saved to {save_path}")
        else:
            plt.show()

    def joint_probability(self, values):
        """
        Compute the joint probability P(values) for a set of node values.
        
        Args:
            values (dict): A dictionary mapping each node to its value (0 or 1).
                           e.g., {'P1': 1, 'P2': 1, 'P3': 0, 'P4': 1}
        """
        prob = 1.0
        
        for node in nx.topological_sort(self.graph):
            if node not in values:
                log.error(f"Missing value for node {node} in joint probability calculation.")
                return 0.0
                
            node_value = values[node]
            parents = list(self.graph.predecessors(node))
            
            try:
                if not parents:
                    prob *= self.cpts[node][node_value]
                else:
                    parent_values = tuple(values[p] for p in parents)
                    prob *= self.cpts[node][parent_values][node_value]
            except KeyError as e:
                log.error(f"Error accessing CPT for node {node}. Missing key: {e}")
                log.error(f"Parents: {parents}, Parent values: {parent_values}, Node value: {node_value}")
                return 0.0
                
        return prob

    def compute_posterior_enumeration(self, query, evidence):
        """
        Compute the posterior probability P(query | evidence) using
        brute-force enumeration.
        
        Args:
            query (dict): e.g., {'P1': 1}
            evidence (dict): e.g., {'P2': 1, 'P3': 0}
        """
        log.info(f"Computing P({query} | {evidence}) via Enumeration...")
        
        # Identify all variables
        all_vars = self.nodes
        hidden_vars = [v for v in all_vars if v not in query and v not in evidence]
        
        numerator = 0.0
        denominator = 0.0
        
        # Iterate over all 2^N assignments for hidden vars
        for hidden_vals in product([0, 1], repeat=len(hidden_vars)):
            hidden_assignment = dict(zip(hidden_vars, hidden_vals))
            
            # Combine assignments
            full_assignment = {**hidden_assignment, **evidence}
            
            # Calculate P(evidence)
            prob_evidence = self.joint_probability(full_assignment)
            denominator += prob_evidence
            
            # Check if this assignment also satisfies the query
            if all(full_assignment[q_var] == q_val for q_var, q_val in query.items()):
                numerator += prob_evidence
                
        if denominator == 0:
            log.warning("Denominator is zero. Evidence may have zero probability.")
            return 0.0
            
        posterior_prob = numerator / denominator
        log.info(f"P({query} | {evidence}) = {posterior_prob:.6f}")
        return posterior_prob

    def variable_elimination(self, query_var, evidence):
        """
        Perform exact inference using variable elimination.
        
        Args:
            query_var (str): The single query variable.
            evidence (dict): e.g., {'P2': 1, 'P3': 0}
            
        Returns:
            Factor: A normalized factor over the query variable.
        """
        log.info(f"Computing P({query_var} | {evidence}) via Variable Elimination...")
        
        factors = []
        
        # 1. Initialize factors from CPTs and apply evidence
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            variables = parents + [node]
            
            # Create CPT dict for the Factor class
            cpt_dict = {}
            if not parents:
                cpt_dict[(0,)] = self.cpts[node][0]
                cpt_dict[(1,)] = self.cpts[node][1]
            else:
                for parent_vals in product([0, 1], repeat=len(parents)):
                    for node_val in [0, 1]:
                        key = tuple(list(parent_vals) + [node_val])
                        cpt_dict[key] = self.cpts[node][parent_vals][node_val]

            factor = Factor(variables, cpt_dict)

            # Apply evidence if this node is in evidence
            if node in evidence:
                factor = reduce_factor(factor, node, evidence[node])
                
            factors.append(factor)

        # 2. Determine elimination order (all nodes not in query or evidence)
        elimination_vars = [v for v in self.nodes if v != query_var and v not in evidence]
        
        # 3. Sum out each variable
        for var in elimination_vars:
            log.debug(f"Eliminating variable: {var}")
            factors_with_var = [f for f in factors if var in f.variables]
            factors_without_var = [f for f in factors if var not in f.variables]
            
            if not factors_with_var:
                continue
                
            # Multiply all related factors
            big_factor = factors_with_var[0]
            for i in range(1, len(factors_with_var)):
                big_factor = multiply_factors(big_factor, factors_with_var[i])
            
            # Sum out the variable
            new_factor = sum_out(big_factor, var)
            
            # Add the new factor back
            factors = factors_without_var + [new_factor]

        # 4. Multiply remaining factors (should only involve query var)
        final_factor = factors[0]
        for i in range(1, len(factors)):
            final_factor = multiply_factors(final_factor, factors[i])
            
        # 5. Normalize
        normalized_result = normalize_factor(final_factor)
        log.info(f"Result for P({query_var} | {evidence}): {normalized_result.cpt}")
        return normalized_result
