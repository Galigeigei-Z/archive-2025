# A Robust Surrogate–Metaheuristic Framework for Black-Box Process Optimization
Non-Equispaced Discrete Cosine Transform with Modified Particle Swarm Optimization
# Overview

This repository contains the Python implementation of the framework proposed in our paper:

“A Robust Surrogate–Metaheuristic Framework for Black-Box Process Optimization:
Non-Equispaced Discrete Cosine Transform with Modified Particle Swarm Optimization.”

The proposed framework integrates two main components:

Non-Equispaced Discrete Cosine Transform (NDCT) for efficient surrogate modeling on irregular and noisy data.

Modified Particle Swarm Optimization (PSO) with adaptive and hybrid strategies for robust global optimization.

This approach is developed for black-box process optimization, particularly within chemical and process engineering applications.

# Key Features
NDCT Surrogate Modeling

Directly handles irregular sample points without pre-interpolation.

Achieves computational complexity comparable to the Fast Fourier Transform (FFT).

Incorporates spectral truncation and Tikhonov regularization for improved robustness against noise.

Modified Particle Swarm Optimization (PSO)

Employs adaptive inertia and coefficient scheduling to balance exploration and exploitation.

Hybridized with Differential Evolution (DE) to enhance population diversity.

Mitigates premature convergence and stagnation issues commonly observed in standard PSO.

# Benchmark Tests and Case Studies

Levy and Rosenbrock benchmark functions (1D and 2D).

Autothermal Reformer (AR) process optimization implemented using the IDAES framework.

Installation
# Clone this repository
git clone https://github.com/Galigeigei-Z/archive-2025.git
cd archive-2025

# Install dependencies
pip install -r requirements.txt
