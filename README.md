# Bayesian Network Inference from Scratch

A Python implementation of Bayesian Networks from scratch, featuring exact inference (Variable Elimination) and approximate inference algorithms (Rejection Sampling, Gibbs Sampling, and Likelihood Weighting). This project was developed for a graduate-level Generative Models course (M.S. in Computer Science).

## Features

* **Bayesian Network Class:** Build, modify, and visualize directed acyclic graphs (DAGs).
* **Exact Inference:** Calculates exact probabilities using **Variable Elimination** and brute-force enumeration.
* **Approximate Inference:** Estimates probabilities using MCMC and other sampling methods:
    * Rejection Sampling
    * Gibbs Sampling
    * Likelihood Weighting
* **Performance Analysis:** Includes a comprehensive script to compare the accuracy (vs. ground truth) and execution time of all sampling algorithms.
* **Modular & Robust:** Code is refactored into a clean `src/` directory with logging, error handling, and a robust `Factor` class for stable inference.

## Core Concepts & Techniques

* **Generative Models:** Bayesian Networks as a foundational generative model.
* **Probabilistic Graphical Models (PGMs):** Representing conditional independence in a graph.
* **Conditional Probability Tables (CPTs):** Quantifying the relationships between variables.
* **Exact Inference:**
    * **Variable Elimination:** An efficient algorithm for exact inference by summing out variables and manipulating factors.
* **Approximate Inference (Sampling):**
    * **Rejection Sampling:** A simple method that generates samples from the prior and rejects those inconsistent with evidence.
    * **Likelihood Weighting:** A more efficient method that fixes evidence variables and weights samples by their likelihood.
    * **Gibbs Sampling (MCMC):** A Markov Chain Monte Carlo method that samples from the Markov blanket of each variable.

---

## How It Works

The project is structured into two main parts: the core library in `src/` and the executable experiments in `scripts/`.

### 1. Core Logic (`src/`)

* `src/bayesian_network.py`: Contains the main `BayesianNetwork` class. It uses `networkx` to manage the graph structure and stores CPTs in dictionaries. This class also implements the exact inference methods. The `variable_elimination` algorithm is built on a `Factor` class to correctly and efficiently multiply, sum-out, and reduce probability tables.
* `src/sampling.py`: Provides standalone functions for the three sampling algorithms (`rejection_sampling`, `gibbs_sampling`, `likelihood_weighting`). Each function takes a `BayesianNetwork` object, a query, and evidence as input.
* `src/utils.py`: Includes a `setup_logging` function and, most importantly, the `Factor` class and its helper functions (`multiply_factors`, `sum_out`, `reduce_factor`). This class is the engine behind the variable elimination algorithm.

### 2. Experiment Scripts (`scripts/`)

* `scripts/run_inference.py`: A command-line script to build the 4-node example network and compute an exact posterior probability using either Variable Elimination (`--method ve`) or enumeration (`--method enumeration`).
* `scripts/run_sampling.py`: A command-line script to estimate a posterior probability on the example network using one of the three sampling methods.
* `scripts/compare_performance.py`: An advanced script that:
    1.  Builds the standard 'Asia' Bayesian Network.
    2.  Calculates a ground-truth probability using Variable Elimination.
    3.  Runs all three sampling methods with varying sample sizes (e.g., 100, 1000, 10000, 100000).
    4.  Measures the squared error and execution time for each run.
    5.  Saves the results to `results/performance_results.csv` and generates plots for error and time comparison.

---

## Core Algorithms: A Deeper Look

This section provides a brief mathematical and conceptual overview of the core inference algorithms implemented.

### 1. The Chain Rule (Joint Probability)

A Bayesian Network simplifies the calculation of the full joint probability distribution by leveraging conditional independence. The chain rule of probability, applied to a BN, states that the joint probability of any assignment of values $(x_1, \dots, x_n)$ to all variables $(X_1, \dots, X_n)$ is the product of the local conditional probabilities.

$$ P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i | \text{Parents}(X_i)) $$

The `joint_probability` method implements this directly. It iterates through the nodes in topological order, finds the conditional probability of each node's assigned value given its parents' values (from the CPT), and multiplies these probabilities together.

### 2. Variable Elimination (Exact Inference)

Variable Elimination (VE) computes an exact posterior probability $P(Q | e)$ by "summing out" hidden variables (those not in the query $Q$ or evidence $e$) one by one. This avoids the exponential complexity of computing the full joint distribution.

The goal is to compute $P(Q, e) = \sum_{h \in H} P(Q, e, h)$, where $H$ is the set of hidden variables. We can rewrite the full joint probability as a product of factors, where each factor $f_i$ corresponds to a CPT:

$$ P(Q, e) = \sum_{h \in H} \prod_{i} f_i(\text{Vars}_i) $$

VE works by pushing the summations "inward" as far as possible. To eliminate a variable $H_j$, we multiply all factors that involve $H_j$, and then sum $H_j$ out of the resulting product. This creates a new, smaller factor that replaces the old ones.

$$ \sum_{h_j} \prod_{i} f_i = \left( \prod_{k \text{ s.t. } H_j \notin \text{Vars}(f_k)} f_k \right) \left( \sum_{h_j} \prod_{l \text{ s.t. } H_j \in \text{Vars}(f_l)} f_l \right) $$

This process is repeated until only factors involving $Q$ remain. The final result is normalized to get $P(Q | e)$. This is implemented using the `Factor` class, which handles the `multiply_factors` and `sum_out` operations.

### 3. Rejection Sampling (Approximate Inference)

This is the most straightforward sampling method. It estimates $P(Q | e)$ by simulating the network's generative process.

$$ P(Q=q | e) \approx \frac{N(Q=q \text{ and } e)}{N(e)} $$

1.  Initialize counts $N(e) = 0$ and $N(Q, e) = 0$.
2.  For $i=1 \to M$ total samples:
3.  Generate a complete sample $x = (x_1, \dots, x_n)$ from the prior by sampling each $X_i$ in topological order from $P(X_i | \text{Parents}(X_i))$.
4.  **Reject** the sample if it is not consistent with the evidence $e$.
5.  If the sample is *not* rejected (i.e., it matches $e$):
    * Increment $N(e)$.
    * If the sample also matches the query $Q$, increment $N(Q, e)$.
6.  Return $\frac{N(Q, e)}{N(e)}$.

**Problem:** This is extremely inefficient if the evidence $P(e)$ is rare, as the vast majority of samples will be rejected.

### 4. Likelihood Weighting (Approximate Inference)

Likelihood Weighting improves on Rejection Sampling by *forcing* the evidence variables to take their observed values. To compensate, it weights each sample by the likelihood of that evidence occurring.

1.  Initialize total weighted sums $W(e) = 0$ and $W(Q, e) = 0$.
2.  For $i=1 \to M$ total samples:
3.  Initialize sample weight $w = 1.0$ and an empty sample $x$.
4.  For each variable $X_i$ in topological order:
    * If $X_i$ is an evidence variable with value $e_j$:
        * Set $x_i = e_j$.
        * Multiply the weight by the likelihood: $w \leftarrow w \times P(x_i = e_j | \text{Parents}(x_i))$.
    * If $X_i$ is *not* an evidence variable:
        * Sample $x_i$ from $P(X_i | \text{Parents}(x_i))$.
5.  Add the sample's weight to the total: $W(e) \leftarrow W(e) + w$.
6.  If the sample $x$ matches the query $Q$, add its weight to the query total: $W(Q, e) \leftarrow W(Q, e) + w$.
7.  Return $\frac{W(Q, e)}{W(e)}$.

This is far more efficient as no samples are rejected, though performance can still degrade if many samples have near-zero weights.

### 5. Gibbs Sampling (Approximate Inference)

Gibbs Sampling is a Markov Chain Monte Carlo (MCMC) method. It estimates the posterior distribution $P(H | e)$ by starting with a random assignment and iteratively re-sampling each non-evidence variable $H_j$ conditioned on the current values of all other variables (its **Markov Blanket**).

1.  Initialize a state $x$ by fixing evidence variables $e$ and setting non-evidence variables $H$ to random values.
2.  For $i=1 \to M$ (total iterations):
3.  For each non-evidence variable $H_j \in H$:
    * Sample a new value $h'_j$ from $P(H_j | x_{-j})$, where $x_{-j}$ is all other variables.
    * Update the state: $x \leftarrow (x_{-j}, h'_j)$.
4.  After a "burn-in" period (e.g., 100 iterations), start collecting the samples $x$.
5.  Estimate $P(Q=q | e)$ by counting the fraction of collected samples where the query $Q$ is true.

The key step is sampling from $P(H_j | x_{-j})$, which simplifies to sampling from its Markov Blanket ($MB(H_j)$):

$$ P(H_j | x_{-j}) = P(H_j | MB(H_j)) = \alpha \times \underbrace{P(H_j | \text{Parents}(H_j))}_{\text{Node's CPT}} \times \underbrace{\prod_{C_k \in \text{Children}(H_j)} P(c_k | \text{Parents}(C_k))}_{\text{Children's CPTs}} $$

We calculate the unnormalized probability for $H_j=1$ and $H_j=0$ using this formula and then normalize to get the distribution from which to sample $h'_j$.

---

## Project Structure

```
python-bayesian-network-inference/
├── .gitignore                  # Ignores standard Python/IDE files
├── LICENSE                     # MIT License
├── README.md                   # This file
├── requirements.txt            # Project dependencies (networkx, matplotlib, etc.)
├── logs/                       # Directory for log files
├── results/                    # Output directory for plots and CSVs
├── src/                        # Main source code
│   ├── __init__.py
│   ├── bayesian_network.py     # The core BayesianNetwork class and exact inference
│   ├── sampling.py             # All sampling algorithms
│   └── utils.py                # Logging setup and the Factor class for VE
├── scripts/                    # Executable scripts
│   ├── run_inference.py        # Runs exact inference on the example network
│   ├── run_sampling.py         # Runs sampling on the example network
│   └── compare_performance.py  # Runs advanced comparison on 'Asia' network
└── run_experiments.ipynb       # Jupyter Notebook to run all scripts
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/python-bayesian-network-inference.git
    cd python-bayesian-network-inference
    ```

2.  **Set Up the Environment:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    
    pip install -r requirements.txt
    ```
    *Note: `pygraphviz` is optional for a cleaner graph layout but can be difficult to install. The code will fall back to a simpler layout if it's not found.*

3.  **Run Experiments:**
    You can either run the individual scripts from the command line or use the provided Jupyter Notebook.

    **Option A: Run `run_experiments.ipynb` (Recommended)**

    Launch Jupyter and open the notebook to run all experiments sequentially and see the results, including the final comparison plots.
    ```bash
    jupyter notebook run_experiments.ipynb
    ```
    
    **Option B: Run Scripts Manually**
    
    Execute the scripts from your terminal.

    * **Run Exact Inference:**
        ```bash
        # Calculate P(P1 | P2=1, P3=0) using Variable Elimination
        python scripts/run_inference.py --query P1 --evidence "P2:1,P3:0" --method ve
        ```

    * **Run Sampling:**
        ```bash
        # Estimate P(P1=1 | P2=1, P3=0) with 50,000 Likelihood Weighting samples
        python scripts/run_sampling.py --query "P1:1" --evidence "P2:1,P3:0" --method likelihood --samples 50000
        ```
   
    * **Run Performance Comparison:**
        ```bash
        # This will take a minute or two to run
        python scripts/compare_performance.py --steps 5 --max_samples 100000
        ```
        After it finishes, check the `results/` folder for the output CSV and plots.

<!--
3.  **Example Output (Performance Comparison):**

    Running `scripts/compare_performance.py` will generate plots like these, showing how Likelihood Weighting and Gibbs Sampling are more accurate and efficient than basic Rejection Sampling, especially when evidence is rare.
    
    ![Error vs. Sample Size](results/performance_error_comparison.png)
    
    ![Time vs. Sample Size](results/performance_time_comparison.png)
-->

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
