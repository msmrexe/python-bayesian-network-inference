import numpy as np
import logging
from collections import defaultdict
import networkx as nx

log = logging.getLogger(__name__)

def topological_sort(bn):
    """Get nodes in topological order."""
    return list(nx.topological_sort(bn.graph))

def _get_prob_from_cpt(bn, node, sample):
    """Helper to get P(node=1 | parents) from CPT given a partial sample."""
    parents = list(bn.graph.predecessors(node))
    if not parents:
        return bn.cpts[node][1]
    else:
        try:
            parent_values = tuple(sample[p] for p in parents)
            return bn.cpts[node][parent_values][1]
        except KeyError:
            # Parent might not be in sample yet if order is wrong
            log.error(f"Could not find parent value for {node}. Check sample: {sample}")
            return 0.0

def rejection_sampling(bn, query, evidence, num_samples):
    """
    Perform Rejection Sampling to estimate P(query | evidence).
    """
    log.info(f"Running Rejection Sampling for P({query} | {evidence})...")
    
    query_var, query_val = list(query.items())[0]
    
    accepted_samples = 0
    query_matches = 0
    nodes = topological_sort(bn)
    
    for _ in range(num_samples):
        sample = {}
        is_consistent = True
        
        # 1. Generate a sample from the network
        for node in nodes:
            prob_node_1 = _get_prob_from_cpt(bn, node, sample)
            sample[node] = np.random.choice([0, 1], p=[1 - prob_node_1, prob_node_1])
            
            # 2. Check for consistency with evidence
            if node in evidence and sample[node] != evidence[node]:
                is_consistent = False
                break
        
        # 3. If consistent (accepted), count it
        if is_consistent:
            accepted_samples += 1
            if sample[query_var] == query_val:
                query_matches += 1
                
    if accepted_samples == 0:
        log.warning("No samples were accepted. Evidence may be rare. Returning 0.")
        return 0.0, 0.0
        
    acceptance_rate = accepted_samples / num_samples
    posterior_prob = query_matches / accepted_samples
    
    log.info(f"Total samples: {num_samples}, Accepted: {accepted_samples} "
             f"(Rate: {acceptance_rate:.4f})")
    log.info(f"Estimated P({query_var}={query_val} | {evidence}) = {posterior_prob:.6f}")
    
    return posterior_prob, acceptance_rate

def likelihood_weighting(bn, query, evidence, num_samples):
    """
    Perform Likelihood Weighting to estimate P(query | evidence).
    """
    log.info(f"Running Likelihood Weighting for P({query} | {evidence})...")
    
    query_var, query_val = list(query.items())[0]
    
    weighted_query_sum = 0.0
    total_weight = 0.0
    nodes = topological_sort(bn)
    
    for _ in range(num_samples):
        sample = {}
        weight = 1.0
        
        for node in nodes:
            prob_node_1 = _get_prob_from_cpt(bn, node, sample)
            
            if node in evidence:
                # 1. Fix evidence node's value
                node_val = evidence[node]
                sample[node] = node_val
                
                # 2. Update weight
                prob_evidence = prob_node_1 if node_val == 1 else (1 - prob_node_1)
                weight *= prob_evidence
            else:
                # 3. Sample as usual
                sample[node] = np.random.choice([0, 1], p=[1 - prob_node_1, prob_node_1])
        
        total_weight += weight
        if sample[query_var] == query_val:
            weighted_query_sum += weight
            
    if total_weight == 0:
        log.warning("Total weight is zero. Returning 0.")
        return 0.0
        
    posterior_prob = weighted_query_sum / total_weight
    
    log.info(f"Total samples: {num_samples}, Total weight: {total_weight:.4f}")
    log.info(f"Estimated P({query_var}={query_val} | {evidence}) = {posterior_prob:.6f}")
    
    return posterior_prob, 1.0 # Acceptance rate is effectively 100%

def gibbs_sampling(bn, query, evidence, num_samples, burn_in=100):
    """
    Perform Gibbs Sampling to estimate P(query | evidence).
    """
    log.info(f"Running Gibbs Sampling for P({query} | {evidence})...")
    
    query_var, query_val = list(query.items())[0]
    
    # 1. Initialize state
    non_evidence_vars = [node for node in bn.nodes if node not in evidence]
    sample = {}
    
    # Fix evidence
    for var, val in evidence.items():
        sample[var] = val
    # Randomly initialize non-evidence
    for var in non_evidence_vars:
        sample[var] = np.random.choice([0, 1])

    query_matches = 0
    total_samples = 0
    
    for i in range(num_samples + burn_in):
        for node in non_evidence_vars:
            # 2. Sample P(node | Markov Blanket)
            prob_node_1 = _prob_given_markov_blanket(bn, node, sample)
            sample[node] = np.random.choice([0, 1], p=[1 - prob_node_1, prob_node_1])
            
        # 3. Collect sample after burn-in
        if i >= burn_in:
            total_samples += 1
            if sample[query_var] == query_val:
                query_matches += 1
                
    if total_samples == 0:
        return 0.0

    posterior_prob = query_matches / total_samples
    
    log.info(f"Total iterations: {num_samples + burn_in}, Burn-in: {burn_in}")
    log.info(f"Collected {total_samples} samples.")
    log.info(f"Estimated P({query_var}={query_val} | {evidence}) = {posterior_prob:.6f}")
    
    return posterior_prob, 1.0

def _prob_given_markov_blanket(bn, node, sample):
    """
    Calculate P(node=1 | MB(node))
    P(node | MB) = P(node | parents) * PROD_{children} P(child | parents(child))
    We only need the ratio, so we can compute:
    P(node=1, MB) = P(node=1 | parents) * PROD_{children} P(child | parents(child) with node=1)
    P(node=0, MB) = P(node=0 | parents) * PROD_{children} P(child | parents(child) with node=0)
    Then P(node=1 | MB) = P(node=1, MB) / (P(node=1, MB) + P(node=0, MB))
    """
    
    # --- P(node | parents) ---
    parents = list(bn.graph.predecessors(node))
    if not parents:
        prob_node_1_given_parents = bn.cpts[node][1]
    else:
        parent_values = tuple(sample[p] for p in parents)
        prob_node_1_given_parents = bn.cpts[node][parent_values][1]
    prob_node_0_given_parents = 1.0 - prob_node_1_given_parents
    
    # --- PROD_{children} P(child | parents(child)) ---
    prob_children_given_node_1 = 1.0
    prob_children_given_node_0 = 1.0
    
    children = list(bn.graph.successors(node))
    for child in children:
        child_parents = list(bn.graph.predecessors(child))
        child_val = sample[child]
        
        # Get P(child | parents with node=1)
        parent_values_1 = tuple(sample[p] if p != node else 1 for p in child_parents)
        prob_child_1 = bn.cpts[child][parent_values_1][child_val]
        prob_children_given_node_1 *= prob_child_1

        # Get P(child | parents with node=0)
        parent_values_0 = tuple(sample[p] if p != node else 0 for p in child_parents)
        prob_child_0 = bn.cpts[child][parent_values_0][child_val]
        prob_children_given_node_0 *= prob_child_0

    # Combine
    unnormalized_prob_1 = prob_node_1_given_parents * prob_children_given_node_1
    unnormalized_prob_0 = prob_node_0_given_parents * prob_children_given_node_0
    
    total = unnormalized_prob_1 + unnormalized_prob_0
    if total == 0:
        return 0.5 # Avoid division by zero, though this state should be impossible
        
    return unnormalized_prob_1 / total
