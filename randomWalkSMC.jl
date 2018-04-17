module randomWalkSMC

using StatsBase
using Distributions
using LightGraphs
using Utils

"""
TO DO:
? extend to multigraphs (involves keeping an edge weight list)
^ SMC implementation
  ++ particle type, etc
  ++ propogation function, etc


"""

immutable struct SamplerState{TG<:AbstractGraph,TI<:Int,T<:AbstractFloat,TS<:AbstractFloat}
  # structure for keeping track of overall sampler quantities

  G_data::TG # observed graph, fixed
  n_edges_data::TI # number of edges in `G_data`, fixed
  size_bias::Bool # whether root vertex is sampled from size-biased distribution, fixed
  a_α::T # hyper-parameter for alpha, fixed
  b_α::T # hyper-parameter for alpha, fixed
  a_λ::T # hyper-parameter for lambda, fixed
  b_λ::T # hyper-parameter for lambda, fixed
  α::Vector{T} # current value of alpha, updates with sampler
  λ::Vector{T} # current value of lambda, updates with sampler
  I::Vector{TS} # current vector of coin flips, updates with sampler
  K::Vector{TS} # current vector of r.w. lengths, updates with sampler
  G_cSMC::ParticleState # current labeled graph and SMC information, updates with sampler
  resample_method::String # either `multinomial` or `residual`

end

mutable struct ParticleState{S <: Int, T <: AbstractGraph, FW <: AbstractFloat, FD <: AbstractFloat}
  # structure for keeping track of state information for particle states
  # the type S is inherited from graph

  graph::T  # graph object (from LightGraphs)
  edge_list::Vector{Tuple{S,S}} # edges in order of insertion (pre-allocate?)
  log_w_list::Vector{FW} # unnormalized log-importance weights corresponding to edge_list proposals
  degrees::Vector{FD} # degree vector ?? is this needed ??
  vertex_map::Vector{S}  # vector where `v[j]` maps vertex `j` in the current graph to `v[j]` of the observed (final) graph
  vertex_map_hoods::Vector{Array{S,1}} # vector of neighborhoods in `G_data` of each vertex in `vertex_map`
  # eligible_edges::Vector{Tuple{S,S}} # vector of edge tuples indicating which edges in the final graph haven't been inserted yet
  # eligible_vertices::Vector{S} # vector of vertex indices indicating which vertices in the final graph haven't been inserted yet
  eigen_system::Base.LinAlg.Eigen  # eigen system of normalized Laplacian of graph

end

immutable struct ParticleSet{T <: Int, TF <: AbstractFloat, S <: Int, GInt <: Int}
  # structure for keeping track of particle weights and states

  n_particles::T  # number of particles
  log_w::Vector{TF}  # unnormalized log-importance weights for SMC
  log_p::Vector{TF}  # log-prob from prior
  unique_state_id::Vector{S}  # pointers to corresponding state in `uniqueParticleStates` vector
  proposals::Vector{Tuple{GInt,GInt}} # vector of edge tuples for current proposals
  # proposals_map::Vector{Tuple{GInt,GInt}} # proposals mapped to vertex set of `G_data`
  ess::String # type of effective sample size to use
  # immortal_particles::Vector{T} # vector of particle indices to condition on surviving resampling (typically `[1]`)

end

function SampleEdgeProposal(p_state::ParticleState, n_samples<:Int, s_state<:SamplerState)
  # function generates `n_samples` edge proposals for `p_state`
  # posterior predictive probs
  t_step = ne(p_state.graph)
  P_I_1 = (s_state.a_α + sum(s_state.I) - s_state.I[t_step])/(s_state.a_α + s_state.b_α + nEdgesFinal - 1);
  a_lambdaPrime = (s_state.a_λ + sum(s_state.K) - s_state.K[t_step]);
  b_lambdaPrime = 1/(s_state.b_λ + s_state.n_edges_data);

  # determine which vertices can have a new vertex attached to them
  eligible_roots = [ length(p_state.vertex_map_hoods[i]) > 0 for i in 1:length(p_state.vertex_map_hoods) ]

  # generate mask for edge proposals
  induced_g, induced_vmap = induced_subgraph(s_state.G_data,p_state.vertex_map) # induced subgraph of `G_data` on vertices in current graph
  assert( induced_vmap == p_state.vertex_map ) # make sure this behaves as expected
  mask = adjacency_matrix(induced_g) # edges also in `p_state.graph` will create a new vertex

  # calculate r.w. probs (negative binomial conditional predictive distribution)
  eig_pgf = ((1 - b_lambdaPrime)^(a_lambdaPrime)).*( (1 - p_state.eigen_system.values).*((1 - b_lambdaPrime.*(1 - p_state.eigen_system.values)).^(-a_lambdaPrime)) );
  eig_pgf[ isinf(eig_pgf) ] = zero(eltype(eig_pgf)) # moore-penrose pseudoinverse
  W = (1 - P_I_1)*Diagonal(p_state.degrees.^(-1/2)) * p_state.eigen_system.vectors * Diagonal(eig_pdf) * p_state.eigen_system.vectors' * Diagonal(p_state.degrees.^(1/2))
  # rwProbs = diag(degrees.^(-1/2)) * U * diag(eigenValues) * U' * diag(degrees.^(1/2));
  elMax!(W, 0) # for numerical stability
  elMin!(W, 1) # for numerical stability

  new_vertex_probs = Diagonal(W) + Diagonal(vec(P_I_1*ones(size(W,1),1)))
  # new_vertex_probs[ eligible_roots .== 0 ] = zero(eltype(W))

  W[ mask .== 0 ] = zero(eltype(W)) # keep only the eligible edges
  for i = 1:size(W,1) # add probability of vertex addition to eligible root vertices
    eligible_roots[i] ? W[i,i] = new_vertex_probs[i] : nothing
  end # there's probably a better way to do this but I can't find an easy way to access all of the diagonal elements of W

  s_state.size_bias ? scale!(p_state.degrees ./ sum(p_state.degrees), W) : nothing # scale by start vertex weights
  v_weights = squeeze(sum(W,2),2)
  log_total_weight = log(sum(v_weights))
  log_total_weight += s_state.size_bias ? -log(sum(p_state.degrees)) : -log(size(W,1)) # normalize root vertex sampling

  root_vertex = wsample(1:length(v_weights), v_weights, n_samples)
  end_vertex = [ wsample(1:length(v_weights), W[root_vertex[j],:]) for j in 1:n_samples ]

  new_vertex = [ root_vertex[j]==end_vertex[j] || has_edge(p_state.graph,root_vertex[j],end_vertex[j]) for j in 1:n_samples ]
  edge_proposals = [ (root_vertex[j], new_vertex[j]==1 ? nv(p_state.graph) + 1 : end_vertex[j]) for j in 1:n_samples ]
  # edge_proposals_map = [ (p_state.vertex_map[root_vertex[j]], new_vertex[j]==0 ? p_state.vertex_map[end_vertex[j]] : sampleNewVertex(p_state.vertex_map_hoods[root_vertex[j]]) ) for j in 1:n_samples ]

  # calculate probability of proposals under model (prior)
  log_p = [ GetLogP(edge_proposals[j], new_vertex[j], p_state, s_state) for j in 1:n_samples ]

  # account for vertex labels of new vertices in proposal
  log_w = -( - log_total_weight + [ new_vertex[j]==1 ? -log(length(p_state.vertex_map_hoods[root_vertex[j]])) : zero(eltype(log_p)) for j in 1:n_samples ] )

  return edge_proposals,log_p,log_w

end

function sampleNewVertex{T<:Int}(hoods::Array{T,1})
  new_vertex = hoods[rand(1:length(hoods))]
end

function GetLogP{T<:Int,TF<:AbstractFloat}(new_edge:Tuple{T},new_vertex::Bool,W::Array{TF,2},p_state::ParticleState,s_state::SamplerState)
  # extract model (prior) probability of `new_edge` from `W`

  log_p = s_state.size_bias ? -log(2*ne(p_state.graph)) : -log(nv(p_state.graph)) # normalize root vertex sampling
  if( !new_vertex )
    log_p += log( W[new_edge[1],new_edge[2]] + W[new_edge[2],new_edge[1]] )
  else
    log_p += log( sum( W[new_edge[1], full(adjacency_matrix(p_state.graph)[new_edge[1],:]) .> 0 ] ) + W[new_edge[1],new_edge[1]] )
    log_p += -log( nv(s_state.G_data) - nv(p_state.graph) )  # for the uniform prior on the mapped labels in G_data
  end
  return log_p
end

function getEdge(G::AbstractGraph,N::Int)
  edge = collect(edges(G))[N]
  return src(edge),dst(edge)
end

function GenerateProposals!(p_set::ParticleSet, unique_states::Vector{ParticleState}, s_state::SamplerState)
  # function generates proposals for each particle

  # get next edge from conditioned particle (this is particle 1)
  sv_particle_ids = p_set.unique_state_id .== 1
  p_set.proposals[sv_particle_ids] = s_state.G_cSMC.edge_list[ne(unique_states[1].graph)+1] # saved state's next edge
  p_set.proposals_map[sv_particle_ids] = # ditto
  p_set.log_p[sv_particle_ids] = # ditto
  p_set.log_w[sv_particle_ids] = # ditto

  # sample appropriate number of edge proposals for each remaining unique state
  if length(unique_states) > 1
    for st_id = 2:length(unique_states)
      particle_ids = p_set.unique_state_id .== st_id
      p_set.proposals[particle_ids],p_set.log_p[particle_ids],p_set.log_w[particle_ids] = SampleEdgeProposal(unique_states[st_id], sum( particle_ids ), s_state)
      # prop_edges,prop_edges_map,log_p,log_q = SampleEdgeProposal(unique_states[st_id], sum( particle_ids ), s_state)
    end
  end
  return nothing

end

function sampleAncestors(particle_set::ParticleSet, unique_states::Vector{ParticleState}, sampler_state::SamplerState)
  # function samples ancestors
  N = particle_set.n_particles

  if sampler_state.resample_method == "multinomial"
    ancestors = [1; wsample(1:N, logSumExpWeights(particle_set.log_w), N - 1)]
    sampled = true
  elseif sampler_state.resample_method == "residual"
    w = exp.(logSumExpWeightsNorm(particle_set.log_w))
    n_tilde = floor.(w.*(N - 1))
    a_tilde = inverse_rle(1:N,n_tilde)
    w_bar = w - n_tilde./(N-1)
    n_bar = wsample(1:N, w_bar, N-1-sum(n_tilde))
    ancestors = [1; a_tilde[randperm(length(a_tilde))]; n_bar ]
    assert( length(ancestors) == N )
    sampled = true
  end
  return ancestors,sampled

end

function PropagateAndResample!(particle_set::ParticleSet, unique_states::Vector{ParticleState}, sampler_state::SamplerState, ess_threshold<:AbstractFloat=1.0)
  # function takes a set of particles, produces a set of proposals, and resamples if ESS is below the threshold

  # propagate particles
  GenerateProposals!(particle_set,unique_states, sampler_state)

  # resample
  resampled = false
  if( EffectiveSampleSize(particle_set.log_w,particle_set.ess) <= ess_threshold ) # resample step
    sampleAncestors!(particle_set)
    resampled = true
  end

  # update `particle_set` and `unique_states`


  return resampled
end


function ConditionalSMC!()

end





function RandomWalkSimpleGraph(; n_edges::Int=100, β::AbstractFloat=0.5, length_distribution::DiscreteUnivariateDistribution=Poisson(1), sizeBias::Bool=0)::AbstractGraph
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
