# test script for networks

using LightGraphs
# using GraphPlot

G = Graph()
add_vertices!(G, 5)
add_edge!(G, 1, 2)
add_edge!(G, 1, 3)
add_edge!(G, 2, 4)
add_edge!(G, 3, 4)
add_edge!(G, 3, 5)
add_edge!(G, 4, 5)

gplot(G, nodelabel=1:nv(G), edgelabel=1:ne(G))

g = graphfamous("karate")
gplot(g)

# gplot doesn't work in Juno yet

##

import MAT

vars = MAT.matread("dolphinAdj.mat")

file = MAT.matopen("dolphinAdj.mat")
dAdj = MAT.read(file,"G_data")
MAT.close(file)

G = squash(Graph(dAdj))
L = NormalizedLaplacian(G)
deg_G = Vector{Float64}(sum(G,2))

function NormalizedLaplacian(g::Graph)
  # computes L = I - D^{-1/2} A D^{-1/2}
  # returns a matrix of type SparseCSC
  Dup = Diagonal( ( 1./(Vector{Float64}(sum(g,2))) ).^(1/2) )
  L = speye(nv(g)) - Dup * adjacency_matrix(g) * Dup # might use scale! functions here
end

##

using Distributions
using LightGraphs
using Gadfly

# abstract type PositiveDiscreteUnivariateDistribution <: DiscreteUnivariateDistribution

function RandomWalkSimpleGraph(;n_edges::Int=100, β::AbstractFloat=0.5, length_distribution::DiscreteUnivariateDistribution=Poisson(1), sizeBias::Bool=0)::AbstractGraph
  # generates a simple graph from the random walk model with new vertex probability `alpha`
  # and random walk length distribution `length_distribution`
  g = Graph([ 0 1; 1 0 ])
  # n = 1

  coinDist = Bernoulli(β)

  for i=1:(n_edges-1)

    coin = rand(coinDist)
    vweights = sizeBias == 1 ? degree(g) : ones(nv(g))

    stv = wsample(1:nv(g), vweights)  # sample start vertex

    if(Bool(coin)) # add a vertex

      add_vertex!(g)
      add_edge!(g, stv, nv(g))
      add_edge!(g, nv(g), stv)

    else # random walk edge

      K = rand(length_distribution)
      edv = K > 0 ? randomwalk(g, stv, K)[end] : stv

      if edv==stv || has_edge(g, stv, edv) # new vertex
        add_vertex!(g)
        add_edge!(g, stv, nv(g))
        add_edge!(g, nv(g), stv)
      else
        add_edge!(g, stv, edv)
        add_edge!(g, edv, stv)
      end

    end

  end
  return g

end


β = 0.25
λ = 4
walk_length_dist = Poisson(λ)
size_bias = false

gtest = RandomWalkSimpleGraph(n_edges = 100, alpha = β, length_distribution = walk_length_dist, sizeBias=size_bias)

Gadfly.spy(adjacency_matrix(gtest))



##
function inplaceTest!(x::Vector{Int},y::Vector{Int})
  x[1] = 10
  y[3] = 100000
end

x = collect(1:10)
y = collect(1:50)

inplaceTest!(x,y)
