"""The advanced test script to compare all methods."""

import argparse
import logging
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bayesian_network import BayesianNetwork
from src.utils import setup_logging
from src.sampling import rejection_sampling, gibbs_sampling, likelihood_weighting

log = logging.getLogger(__name__)

def create_asia_network():
    """
    Creates the 'Asia' network, a classic small BN.
    A: Visit to Asia?
    S: Smoker?
    T: Tuberculosis?
    L: Lung Cancer?
    B: Bronchitis?
    E: TB or Lung Cancer? (Either)
    X: X-Ray Positive?
    D: Dyspnea (Shortness of breath)?
    """
    log.info("Creating 'Asia' network for performance comparison.")
    bn = BayesianNetwork()
    nodes = ['A', 'S', 'T', 'L', 'B', 'E', 'X', 'D']
    for node in nodes:
        bn.add_node(node)

    # Structure
    bn.add_edge('A', 'T')
    bn.add_edge('S', 'L')
    bn.add_edge('S', 'B')
    bn.add_edge('T', 'E')
    bn.add_edge('L', 'E')
    bn.add_edge('E', 'X')
    bn.add_edge('E', 'D')
    bn.add_edge('B', 'D')

    # CPTs
    bn.set_cpt('A', {0: 0.99, 1: 0.01}) # P(A)
    bn.set_cpt('S', {0: 0.5, 1: 0.5})  # P(S)
    
    # P(T | A)
    bn.set_cpt('T', {(0,): {0: 0.99, 1: 0.01}, (1,): {0: 0.95, 1: 0.05}})
    # P(L | S)
    bn.set_cpt('L', {(0,): {0: 0.99, 1: 0.01}, (1,): {0: 0.9, 1: 0.1}})
    # P(B | S)
    bn.set_cpt('B', {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.4, 1: 0.6}})
    
    # P(E | T, L)
    bn.set_cpt('E', {(0, 0): {0: 1.0, 1: 0.0},
                       (0, 1): {0: 0.0, 1: 1.0},
                       (1, 0): {0: 0.0, 1: 1.0},
                       (1, 1): {0: 0.0, 1: 1.0}}) # E is T or L
                       
    # P(X | E)
    bn.set_cpt('X', {(0,): {0: 0.95, 1: 0.05}, (1,): {0: 0.02, 1: 0.98}})
    
    # P(D | E, B)
    bn.set_cpt('D', {(0, 0): {0: 0.9, 1: 0.1},
                       (0, 1): {0: 0.2, 1: 0.8},
                       (1, 0): {0: 0.3, 1: 0.7},
                       (1, 1): {0: 0.1, 1: 0.9}})
                       
    bn.draw_network("results/asia_network.png")
    return bn

def plot_results(df, exact_prob, query_str):
    """Plots error and time comparisons and saves them."""
    
    # Error Plot
    plt.figure(figsize=(12, 6))
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        plt.plot(method_df['samples'], method_df['error'], 'o-', label=method)
    
    plt.title(f'Sampling Error vs. Sample Size for P({query_str})')
    plt.xlabel('Number of Samples')
    plt.ylabel(f'Squared Error (Exact = {exact_prob:.6f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/performance_error_comparison.png')
    log.info("Saved error comparison plot to results/performance_error_comparison.png")

    # Time Plot
    plt.figure(figsize=(12, 6))
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        plt.plot(method_df['samples'], method_df['time'], 'o-', label=method)
        
    plt.title(f'Execution Time vs. Sample Size for P({query_str})')
    plt.xlabel('Number of Samples')
    plt.ylabel('Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/performance_time_comparison.png')
    log.info("Saved time comparison plot to results/performance_time_comparison.png")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Compare Sampling Method Performance")
    parser.add_argument(
        "--steps", 
        type=int, 
        default=5, 
        help="Number of sample sizes to test."
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=100000, 
        help="Maximum number of samples to test."
    )
    args = parser.parse_args()

    # Create results dir
    if not os.path.exists("results"):
        os.makedirs("results")

    # 1. Setup Network and Ground Truth
    bn = create_asia_network()
    
    # Query: P(LungCancer=1 | Dyspnea=1, VisitToAsia=0)
    query_var = 'L'
    query = {'L': 1}
    evidence = {'D': 1, 'A': 0}
    query_str = "L=1 | D=1, A=0"
    
    log.info(f"Calculating exact ground truth for P({query_str})...")
    exact_factor = bn.variable_elimination(query_var, evidence)
    exact_prob = exact_factor.cpt[(1,)]
    log.info(f"Exact Probability P({query_str}) = {exact_prob:.6f}")
    
    # 2. Setup sample sizes to test
    sample_sizes = np.logspace(
        2, np.log10(args.max_samples), args.steps, dtype=int
    )
    methods = {
        "Rejection": rejection_sampling,
        "Likelihood": likelihood_weighting,
        "Gibbs": gibbs_sampling
    }
    
    results = []
    
    # 3. Run all methods for all sample sizes
    for n in sample_sizes:
        print(f"\n--- Testing with {n} samples ---")
        for name, func in methods.items():
            log.info(f"Running {name}...")
            start_time = time.time()
            
            prob, rate = func(bn, query, evidence, n)
            
            time_taken = time.time() - start_time
            error = (prob - exact_prob) ** 2
            
            results.append({
                "method": name,
                "samples": n,
                "probability": prob,
                "error": error,
                "time": time_taken,
                "acceptance_rate": rate if name == "Rejection" else np.nan
            })
            log.info(f"Finished {name}. Prob: {prob:.6f}, Error: {error:.2e}, Time: {time_taken:.4f}s")
    
    # 4. Process and plot results
    df = pd.DataFrame(results)
    print("\n--- Performance Comparison Results ---")
    print(df.to_string())
    
    # Save to CSV
    df.to_csv("results/performance_results.csv", index=False)
    log.info("Saved full results to results/performance_results.csv")
    
    # Plot
    plot_results(df, exact_prob, query_str)
    print("\nComparison plots saved to 'results/' directory.")
    
if __name__ == "__main__":
    main()
