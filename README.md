# SMC and Particle Gibbs for random walk models of networks

This repo contains code for Sequential Monte Carlo and Particle Gibbs algorithms for inference in random walk models of networks, as formulated in ["Random Walk Models of Network Formation and Sequential Monte Carlo Methods for Graphs"](https://arxiv.org/abs/1612.06404) by B. Bloem-Reddy and P. Orbanz (accepted for publication in JRSSB).

See DEMO.jl for basic usage (to be run interactively via the Julia REPL). The main inference algorithms represent a network as an edge list, but utility functions are provided for converting between adjacency matrices and edge lists.

Please note that the inference algorithms are memory intensive: linear in the number of particles, each of which is of size  O(\# edges)^3. The reason for this is computational efficiency; the same eigenvalue decomposition is required multiple times for each time step, so it is computed once and stored, rather than re-computing each time it is needed. (The eigenvalue decomposition is the most computationally expensive (by far) part of the algorithm.) All memory is pre-allocated.

To do:
  - write R wrappers
  - extend to multigraphs
  - (longer term) allow for user-defined models
