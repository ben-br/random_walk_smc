# functions for random walk SMC
using StatsBase

include "Utils.jl"

# abstract type EdgeList <: Array{Int64,2} end # stores edges in a graph
abstract type VertexMap <: Vector{Int64}
abstract type ParticlePath <: Vector end # stores retained smc particle path

immutable struct SamplerState{TG::Array{Int64,2},TE<:Real,T<:AbstractFloat}
  # structure for keeping track of overall sampler quantities

  data_elist::TG # observed graph, fixed, edge list indices
  data_elist_vals::Vector{TE} # observed graph, fixed, edge list values (e.g., 1 for a simple graph)
  degree_data::Vector{Int64}
  ne_data::Int64 # number of edges in `G_data`, fixed
  nv_data::Int64 # number of edges in `G_data`, fixed
  size_bias::Bool # whether root vertex is sampled from size-biased distribution, fixed
  a_α::T # hyper-parameter for alpha, fixed
  b_α::T # hyper-parameter for alpha, fixed
  a_λ::T # hyper-parameter for lambda, fixed
  b_λ::T # hyper-parameter for lambda, fixed
  α::Vector{T} # current value of alpha, updates with sampler
  λ::Vector{T} # current value of lambda, updates with sampler
  I::Vector{Int64} # current vector of coin flips, updates with sampler
  K::Vector{Int64} # current vector of r.w. lengths, updates with sampler
  G_cSMC::ParticlePath # current labeled graph and SMC information, updates with sampler
  resample_method::String # either `multinomial` or `residual`
  resample_ess_threshold::Float64

end

function new_sampler_state(G_data::Vector{Int64},)


end

function new_sampler_state(G_data::Array{Int64}{2},)


end

struct ParticleState
  # structure for keeping track of state information for particle states

  n_edges::Int64 # how many edges have been inserted
  edge_list::Array{Int64,2} # edges in order of insertion (pre-allocate)
  n_vertices::Vector{Int64}
  new_vertex::Vector{Bool}
  # log_p::Vector{FW} # unnormalized log-importance weights corresponding to edge_list proposals
  degrees::Vector{Int64} # degree vector ?? is this needed ??
  vertex_map::Vector{Int64}  # vector where `v[j]` maps vertex `j` in the current graph to `v[j]` of the observed (final) graph
  vertex_unmap::Vector{Int64} # vector where v[j] maps vertex j in the final graph to v[j] in the current graph
  edge_queue::Array{Int64}{2} # vector of edge tuples indicating which edges in the final graph haven't been inserted yet
  # eligible_vertices::Vector{S} # vector of vertex indices indicating which vertices in the final graph haven't been inserted yet
  # eigen_system::Base.LinAlg.Eigen  # eigen system of normalized Laplacian of graph

end

function initialize_blank_particle_state(n_edges::Int64,n_edges_max::Int64,n_vertices_max::Int64)

  ps = ParticleState(n_edges,
                      zeros(Int64,n_edges,2), # edge_list
                      zeros(Int64,1), # n_vertices
                      [false], # new_vertex
                      zeros(Int64,n_vertices_max), # degrees
                      zeros(Int64,n_vertices_max), # vertex_map
                      zeros(Int64,n_vertices_max), # vertex_unmap
                      zeros(Bool,n_edges_max,2) # edge_queue
                      )
  return ps
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


function rw_smc(n_particles::Int64,s_state::SamplerState,resample_ess_threshold::Float64)

  T = s_state.ne_data
  nv_max = s_state.nv_data

  # initialize/pre-allocate particle paths
  particle_container = Array{Array{ParticleState,1},1}(n_particles)
  for p in 1:n_particles
    particle_container[p] = [initialize_blank_particle_state(t,T,nv_max) for t in 1:T]
  end

  # sample t=1
  t = 1
  edge_idx = sample(1:T,n_particles)
  for p in 1:n_particles
    particle_container[p][1].edge_list[1,:] = s_state.data_elist[edge_idx[p],:] # add new edge
    particle_container[p][1].n_vertices[:] = 2 # update number of vertices
    particle_container[p][1].degrees[1:2] = 1 # update degrees
    particle_container[p][1].vertex_map[1:2] = s_state.data_elist[edge_idx[p],:] # update vertex map
    particle_container[p][1].vertex_unmap[s_state.data_elist[edge_idx[p],:]] = [1,2] # update vertex unmap
    # construct initial edge queue
    eq = [ ( in(s_state.data_elist[m,1],s_state.data_elist[edge_idx[p],:]) || in(s_state.data_elist[m,2],s_state.data_elist[edge_idx[p],:]) ) for m in 1:T ]
    particle_container[p][1].edge_queue[:] = eq
  end

  log_weights = -log(n_particles)
  for t in 2:T

    # generate exhaustive proposal set
    n_edges_in_queue = zeros(Int64,n_particles)
    for p = 1:n_particles
      n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
    end
    cumu_eq = cumsum(n_edges_in_queue)
    total_eq = n_edges_in_queue[end]

    edge_proposals = zeros(Int64,total_eq)
    edge_logp = zeros(Float64,total_eq)
    for p = 1:n_particles
      p==1 ? idx = 1:cumu_eq[p] : idx = (cumu_ep[p-1]+1):cumu_eq[p]
      edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
      edge_logp[idx] = generateProposalProbsRW(particle_container[p][t-1],s_state)
    end

    # sample edges for propogation
    edge_samples = stratifiedResample(logSumExpWeights(edge_logp),n_particles)

    # keep track of ancestors
    ancestors[t,:] = getAncestors(edge_samples,cumu_eq)

    # update particles
    updateParticles!(particle_container,t,edge_samples,ancestors)

  end

  # update_particles!(particle_container,t,edge_idx)


end

function getAncestors(es::Array{Int64,1},ceq::Array{Int64,1})

  anc = zeros(Int64,length(es))
  for i = 1:length(es)
    anc[i] = findlast( es[i] .<= ceq )
  end
  return anc
end

function updateParticles!(particle_container::Array{Array{ParticleState,1},1},t::Int64,s_state::SamplerState,sampled_edges::Array{Int64,2},ancestors::Vector{Int64})
"""
  Updates particle_container at step t to append newly sampled edges
"""
  if t==1
    for p = 1:length(particle_container)
      particle_container[p][1].edge_list[1,:] = s_state.data_elist[edge_idx[p],:] # add new edge
      particle_container[p][1].n_vertices[:] = 2 # update number of vertices
      particle_container[p][1].new_vertex[:] = true # update new_vertex flag
      particle_container[p][1].degrees[1:2] = 1 # update degrees
      particle_container[p][1].vertex_map[1:2] = s_state.data_elist[edge_idx[p],:] # update vertex map
      particle_container[p][1].vertex_unmap[s_state.data_elist[edge_idx[p],:]] = [1,2] # update vertex unmap
      # construct initial edge queue
      eq = [ ( in(s_state.data_elist[m,1],s_state.data_elist[edge_idx[p],:]) || in(s_state.data_elist[m,2],s_state.data_elist[edge_idx[p],:]) ) for m in 1:T ]
      particle_container[p][1].edge_queue[:] = eq
    end

  else
    T = length(particle_container[1])

    particle_container[p][t].edge_list[1:(t-1),:] = particle_container[ancestors[p]][t-1].edge_list[1:(t-1),:] # copy previous edges
    particle_container[p][t].edge_list[t,:] = s_state.data_elist[sampled_edges[p],:] # add new edge

    particle_container[p][t].vertex_map[:] = particle_container[ancestors[p]][t-1].vertex_map[:] # copy previous vertex_map
    new_vertex = [!in(s_state.data_elist[sampled_edges[p],1],particle_container[p][t].vertex_map), !in(s_state.data_elist[sampled_edges[p],2],particle_container[p][t].vertex_map) )]
    particle_container[p][t].new_vertex[:] = (new_vertex[1] || new_vertex[2])
    particle_container[p][t].n_vertices = sum(particle_container[p][t].vertex_map > 0) + sum(new_vertex)
    # update vertex_map
    sum(new_vertex)==1 ? particle_container[p][t].vertex_map[particle_container[p][t].n_vertices] = s_state.data_elist[sampled_edges[p],find(new_vertex)] : nothing
    # update vertex_unmap
    sum(new_vertex)==1 ? particle_container[p][t].vertex_unmap[s_state.data_elist[sampled_edges[p],find(new_vertex)]] = particle_container[p][t].n_vertices
    # update edge queue
    particle_container[p][t].edge_queue[:] = particle_container[ancestors[p]][t-1].edge_queue[:]
    if new_vertex # add new edges in the queue
      # mark edges
      vtx = s_state.data_elist[sampled_edges[p],find(new_vertex)]
      edges_to_add = [ in(vtx,s_state.data_elist[m,:] for m=1:T ]
      particle_container[p][t].edge_queue[edges_to_add] = true
    end
    particle_container[p][t].edge_queue[sampled_edges[p]] = false # remove new edge from queue

  end

end

function generateProposalProbsRW(p_state::ParticleState, s_state::SamplerState)
"""
  Returns the log probability under the prior of the entire set of eligible edges,
  as indexed in the enumeration of data edges.
"""

  t_step = p_state.n_edges + 1
  t_remaining = s_state.ne_data - t_step
  # calculate predictive probabiliy of a new vertex
  P_I_1 = (s_state.a_α + sum(s_state.I) - s_state.I[t_step])/(s_state.a_α + s_state.b_α + s_state.ne_data - 1);
  # calculate parameters of predictive distribution for random walk length
  a_lambdaPrime = (s_state.a_λ + sum(s_state.K) - s_state.K[t_step]);
  b_lambdaPrime = 1/(s_state.b_λ + s_state.ne_data);

  # determine which vertices can have a new vertex attached to them
  # eligible_roots = [ p_state.degrees[i] < s_state.degree_data[p_state.vertex_map[i]] for i in 1:p_state.n_vertices ].*1

  # calculate r.w. probs (negative binomial predictive distribution)
  eig_pgf = ((1 - b_lambdaPrime)^(a_lambdaPrime)).*( (1 - p_state.eigen_system.values).*((1 - b_lambdaPrime.*(1 - p_state.eigen_system.values)).^(-a_lambdaPrime)) );
  eig_pgf[ isinf(eig_pgf) ] = zero(eltype(eig_pgf)) # moore-penrose pseudoinverse
  W = Diagonal(p_state.degrees.^(-1/2)) * p_state.eigen_system.vectors * Diagonal(eig_pgf) * p_state.eigen_system.vectors' * Diagonal(p_state.degrees.^(1/2))
  # rwProbs = diag(degrees.^(-1/2)) * U * diag(eigenValues) * U' * diag(degrees.^(1/2));
  elMax!(W, 0) # for numerical stability
  elMin!(W, 1) # for numerical stability

  n_eligible = sum(p_state.edge_queue)
  log_p = zeros(Float64,n_eligible)
  # log_q = zeros(Float64,n_eligible)
  # edge_proposal_idx = find(p_state.eligible)
  for n in 1:n_eligible
    ed = [s_state.data_elist[edge_proposal_idx[n],1]]; s_state.data_elist[edge_proposal_idx[n],2]]
    if ed[1] > 0 && ed[2] > 0 # vertices not yet inserted have index 0 in current particle state
      log_p[n] = log(1 - P_I_1) + log(W[ed[1],ed[2]] + W[ed[2],ed[1]])
    else
      log_p[n] = log(P_I_1) - log(t_remaining) # second term accounts for the "random permutation" applied to obtain the observed labels
    end
  end

  return log_p
end
