import argparse
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bayesian_network import BayesianNetwork
from src.utils import setup_logging

log = logging.getLogger(__name__)

def create_example_network():
    """Creates the 4-node example network from the notebook."""
    bn = BayesianNetwork()
    nodes = ["P1", "P2", "P3", "P4"]
    for node in nodes:
        bn.add_node(node)

    bn.add_edge("P1", "P2")
    bn.add_edge("P2", "P3")
    bn.add_edge("P2", "P4")

    # CPTs
    P1 = {0: 0.6, 1: 0.4}
    P2 = {(0,): {0: 0.5, 1: 0.5}, (1,): {0: 0.2, 1: 0.8}}
    P3 = {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.8, 1: 0.2}}
    P4 = {(0,): {0: 0.5, 1: 0.5}, (1,): {0: 0.2, 1: 0.8}}

    bn.set_cpt('P1', P1)
    bn.set_cpt('P2', P2)
    bn.set_cpt('P3', P3)
    bn.set_cpt('P4', P4)
    
    log.info("Example network created.")
    bn.draw_network("results/example_network.png")
    return bn

def parse_dict_str(s):
    """Parses a string 'k1:v1,k2:v2' into a dict {k1: v1, k2: v2}."""
    if not s:
        return {}
    try:
        return {item.split(':')[0]: int(item.split(':')[1]) for item in s.split(',')}
    except Exception as e:
        log.error(f"Could not parse query/evidence string: {s}. Error: {e}")
        log.error("Expected format: 'Node1:1,Node2:0'")
        sys.exit(1)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Exact Inference on Bayesian Network")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="Query variable. e.g., 'P1'"
    )
    parser.add_argument(
        "--evidence", 
        type=str, 
        default="", 
        help="Evidence variables and values. e.g., 'P2:1,P3:0'"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["ve", "enumeration"], 
        default="ve", 
        help="Inference method to use."
    )
    
    args = parser.parse_args()

    # Create results dir
    if not os.path.exists("results"):
        os.makedirs("results")

    bn = create_example_network()
    
    evidence = parse_dict_str(args.evidence)
    query_var = args.query

    if args.method == "ve":
        result_factor = bn.variable_elimination(query_var, evidence)
        print("\n--- Variable Elimination Result ---")
        print(f"P({query_var} | {evidence})")
        for val, prob in result_factor.cpt.items():
            print(f"  {query_var}={val[0]}: {prob:.6f}")
            
    elif args.method == "enumeration":
        # Enumeration method was written to handle full query dict
        query_1 = {query_var: 1}
        prob_1 = bn.compute_posterior_enumeration(query_1, evidence)
        
        print("\n--- Enumeration Result ---")
        print(f"P({query_var} | {evidence})")
        print(f"  {query_var}=1: {prob_1:.6f}")
        print(f"  {query_var}=0: {1.0 - prob_1:.6f}")

if __name__ == "__main__":
    main()
