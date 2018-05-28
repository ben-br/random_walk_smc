# random graph generating functions

using Distributions

function randomWalkSimpleGraph(;n_edges::Int=100, alpha_prob::AbstractFloat=0.5,length_distribution::DiscreteUnivariateDistribution=Poisson(1), sizeBias::Bool=false)
  # generates a simple graph from the random walk model with new vertex probability β
  # and random walk length distribution `length_distribution`
  g = zeros(Int64,n_edges,2)
  g[1,:] = [1 2] # first edge
  n = 1
  nv = 2
  deg = [1;1]

  min_offset = 1 - minimum(length_distribution)

  coinDist = Bernoulli(alpha_prob)

  for i = 2:n_edges

    coin = rand(coinDist)
    vweights = (sizeBias ? deg : ones(Float64,nv))

    stv = wsample(1:nv, vweights)  # sample start vertex

    if(Bool(coin)) # add a vertex
      nv += 1
      g[i,:] = [stv,nv]
      push!(deg,1)

    else # random walk edge
      K = rand(length_distribution) + min_offset
      edv = (K > 0 ? randomwalk(g[1:i-1,:], stv, K)[end] : stv)

      if edv==stv || has_edge(g[1:i-1,:], stv, edv) # new vertex
        nv += 1
        g[i,:] = [stv,nv]
        push!(deg,1)

      else
        g[i,:] = [stv,edv]
        deg[stv] += 1
        deg[edv] += 1

      end

    end

  end
  return g

end


function has_edge(edgelist::Array{Int64,2},edge::Vector{Int64})
"""
  Checks for the directed edge in edgelist
"""
  return any( [ edge == edgelist[i,:] for i = 1:size(edgelist,1) ] )
end

function has_edge(edgelist::Array{Int64,2},v1::Int64,v2::Int64)
"""
  Checks for the undirected edge [v1, v2] in edgelist
"""
  return any( [ ([v1,v2] == edgelist[i,:] || [v2,v1] == edgelist[i,:]) for i = 1:size(edgelist,1) ] )
end

function randomwalk(edgelist::Array{Int64,2},start_vertex::Int64,len::Int64)

  A = edgelist2adj(edgelist)
  nv = size(A,2)
  vc = start_vertex
  p = zeros(Int64,len+1)
  p[1] = vc
  for k = 1:len
    vn = wsample(1:nv,full(A[:,vc]))
    vc = vn
    p[k+1] = vc
  end
  return p
end
