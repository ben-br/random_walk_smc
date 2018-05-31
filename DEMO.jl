####################
####### DEMO #######
####################

## import required packages ##
# REQUIRE?
using Distributions
import LightGraphs

include("rw_smc.jl")
include("Utils.jl")
include("rand_rw.jl")
include("gibbs_updates.jl")

# set seed
srand(0)

# set synthetic data parameters
const n_edges_data = 50 # number of edges
const α = 0.25 # parameter for new vertex probability
const λ = 4.0 # paramter for random walk length distribution
const ld = Poisson(λ) # random walk length distribution (automatically shifted by +1 if minimum is zero)
const sb = true # degree-biased distribution for first vertex at each step (false for uniform distribution)

# sample simple graph (returns edge list)
g = randomWalkSimpleGraph(n_edges=n_edges_data,alpha_prob=α,length_distribution=ld,sizeBias=sb)

# sample multi-graph (returns edge list)
# g = randomWalkMultiGraph(n_edges=n_edges_data,alpha_prob=α,length_distribution=ld,sizeBias=sb)


######################################
# INTERFACE WITH LIGHTGRAPHS PACKAGE #
######################################

# create LightGraphs.Graph object for use with LightGraphs functions
lg = LightGraphs.Graph(edgelist2adj(g))
# Examples:
lg_conn = LightGraphs.is_connected(lg)
lg_diam = LightGraphs.diameter(lg)
