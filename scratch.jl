# test bed
using Distributions
using Plots
gr()

include("Utils.jl")
include("rw_smc.jl")
include("rand_rw.jl")

import MAT


# load adjacency matrix
file = MAT.matopen("data/dolphinAdj.mat")
dAdj = Int64.(MAT.read(file,"G_data"))
MAT.close(file)

# convert to edge list
dElist = adj2edgelist(dAdj)

L = normalizedLaplacian(dElist)
deg = getDegrees(dElist)

esys = eigfact(Symmetric(full(L)))
esys[:values][1] = 0.0

ld = Poisson(3)
g = randomWalkSimpleGraph(n_edges=500,beta_prob=0.33,length_distribution=ld,sizeBias=true)
