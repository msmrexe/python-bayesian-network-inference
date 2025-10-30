"""Script to run approximate inference."""

import argparse
import logging
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import setup_logging
from src.sampling import rejection_sampling, gibbs_sampling, likelihood_weighting
from scripts.run_inference import create_example_network, parse_dict_str

log = logging.getLogger(__name__)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Sampling Inference on Bayesian Network")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="Query variable and value. e.g., 'P1:1'"
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
        choices=["rejection", "gibbs", "likelihood"], 
        required=True, 
        help="Sampling method to use."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=10000, 
        help="Number of samples or iterations."
    )
    
    args = parser.parse_args()

    bn = create_example_network()
    
    evidence = parse_dict_str(args.evidence)
    query = parse_dict_str(args.query)
    
    if len(query) != 1:
        log.error("Query must be a single variable. e.g., 'P1:1'")
        sys.exit(1)

    print(f"\nRunning {args.method} sampling with {args.samples} samples...")
    print(f"Query: P({query} | {evidence})")
    
    start_time = time.time()
    
    if args.method == "rejection":
        prob, rate = rejection_sampling(bn, query, evidence, args.samples)
        print(f"Acceptance Rate: {rate:.4f}")
        
    elif args.method == "likelihood":
        prob, _ = likelihood_weighting(bn, query, evidence, args.samples)
        
    elif args.method == "gibbs":
        prob, _ = gibbs_sampling(bn, query, evidence, args.samples)

    end_time = time.time()
    
    print("\n--- Sampling Result ---")
    print(f"Estimated Probability: {prob:.6f}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
