# functions for random walk SMC
using StatsBase

include("Utils.jl")

# abstract type EdgeList <: Array{Int64,2} end # stores edges in a graph
# abstract type VertexMap <: Vector end
# abstract type ParticlePath <: Vector end # stores retained smc particle path

struct ParticleState
  # structure for keeping track of state information for particle states

  n_edges::Vector{Int64} # how many edges have been inserted
  edge_list::Array{Int64,2} # edges in order of insertion (pre-allocate)
  n_vertices::Vector{Int64}
  new_vertex::Vector{Bool}
  log_w::Vector{Float64} # unnormalized log-importance weight corresponding to edge_list proposals
  ancestor::Vector{Int64}
  degrees::Vector{Int64} # degree vector ?? is this needed ??
  vertex_map::Vector{Int64}  # vector where `v[j]` maps vertex `j` in the current graph to `v[j]` of the observed (final) graph
  vertex_unmap::Vector{Int64} # vector where v[j] maps vertex j in the final graph to v[j] in the current graph
  edge_queue::Vector{Bool} # vector of categories indicating which edges in the final graph are currently eligible to be inserted (=true)
  # eligible_vertices::Vector{S} # vector of vertex indices indicating which vertices in the final graph haven't been inserted yet
  # eigen_system::Base.LinAlg.Eigen  # eigen system of normalized Laplacian of graph

end

struct SamplerState
  # structure for keeping track of overall sampler quantities

  data_elist::Array{Int64,2} # observed graph, fixed, edge list indices
  # data_elist_vals::Vector{Int64} # observed graph, fixed, edge list values (e.g., 1 for a simple graph)
  degree_data::Vector{Int64}
  ne_data::Vector{Int64} # number of edges in `G_data`, fixed
  nv_data::Vector{Int64} # number of edges in `G_data`, fixed
  size_bias::Vector{Bool} # whether root vertex is sampled from size-biased distribution, fixed
  a_α::Vector{Float64} # hyper-parameter for alpha, fixed
  b_α::Vector{Float64} # hyper-parameter for alpha, fixed
  a_λ::Vector{Float64} # hyper-parameter for lambda, fixed
  b_λ::Vector{Float64} # hyper-parameter for lambda, fixed
  α::Vector{Float64} # current value of alpha, updates with sampler
  λ::Vector{Float64} # current value of lambda, updates with sampler
  I::Vector{Int64} # current vector of coin flips, updates with sampler
  K::Vector{Int64} # current vector of r.w. lengths, updates with sampler
  G_cSMC::Vector{ParticleState} # current labeled graph and SMC information, updates with sampler

end

function new_sampler_state(elist_data::Array{Int64,2},
                            size_bias::Bool,
                            a_α::Float64,b_α::Float64,
                            a_λ::Float64,b_λ::Float64,
                            α::Float64,λ::Float64,
                            I::Vector{Int64},K::Vector{Int64},
                            particle::Array{ParticleState,1})
"""
  Construct new sampler state object from given data and parameters
"""
  return SamplerState(elist_data, # edge list
                      getDegrees(elist_data), # degrees
                      [size(elist_data,1)], # number of edges
                      [maximum(elist_data)], # number of vertices
                      [size_bias],
                      [a_α],[b_α],[a_λ],[b_λ], # hyper-parameters
                      [α],[λ], # parameters
                      I, # I vector
                      K, # K vector
                      particle)

end


function initialize_blank_particle_state(n_edges::Int64,n_edges_max::Int64,n_vertices_max::Int64)

  ps = ParticleState([n_edges],
                      zeros(Int64,n_edges,2), # edge_list
                      zeros(Int64,1), # n_vertices
                      [false], # new_vertex
                      zeros(Float64,1), # log_w
                      zeros(Int64,1), # ancestor
                      zeros(Int64,n_vertices_max), # degrees
                      zeros(Int64,n_vertices_max), # vertex_map
                      zeros(Int64,n_vertices_max), # vertex_unmap
                      falses(n_edges_max) # edge_queue
                      )
  return ps
end

function reset_particle_state!(particle::ParticleState)
"""
  Function to reset the fields of a particle state
"""
  # leave .n_edges unchanged
  particle.edge_list[:] = zero(Int64)
  particle.n_vertices[:] = zero(Int64)
  particle.new_vertex[:] = false
  particle.log_w[:] = zero(Int64)
  particle.ancestor[:] = zero(Int64)
  particle.degrees[:] = zero(Int64)
  particle.vertex_map[:] = zero(Int64)
  particle.vertex_unmap[:] = zero(Int64)
  particle.edge_queue[:] = false
end

function reset_particles!(particle_container::Array{Array{ParticleState,1},1})

  np = length(particle_container)
  ne = length(particle_container[1])

  for p = 1:np
    for t = 1:ne
      reset_particle_state!(particle_container[p][t])
    end
  end

end

function copyParticlelState!(particle_target::ParticleState,particle_source::ParticleState)

  particle_target.n_edges[:] = particle_source.n_edges[:]
  particle_target.edge_list[:] = particle_source.edge_list[:]
  particle_target.n_vertices[:] = particle_source.n_vertices[:]
  particle_target.new_vertex[:] = particle_source.new_vertex[:]
  particle_target.log_w[:] = particle_source.log_w[:]
  particle_target.ancestor[:] = particle_source.ancestor[:]
  particle_target.degrees[:] = particle_source.degrees[:]
  particle_target.vertex_map[:] = particle_source.vertex_map[:]
  particle_target.vertex_unmap[:] = particle_source.vertex_unmap[:]
  particle_target.edge_queue[:] = particle_source.edge_queue[:]

end

function copyParticlelState!(particle_target::ParticleState,max_idx::Int64,particle_source::ParticleState)

  particle_target.n_edges[:] = particle_source.n_edges[:]
  particle_target.edge_list[1:max_idx] = particle_source.edge_list[:]
  particle_target.n_vertices[:] = particle_source.n_vertices[:]
  particle_target.new_vertex[:] = particle_source.new_vertex[:]
  particle_target.log_w[:] = particle_source.log_w[:]
  particle_target.ancestor[:] = particle_source.ancestor[:]
  particle_target.degrees[:] = particle_source.degrees[:]
  particle_target.vertex_map[:] = particle_source.vertex_map[:]
  particle_target.vertex_unmap[:] = particle_source.vertex_unmap[:]
  particle_target.edge_queue[:] = particle_source.edge_queue[:]

end

# struct ParticleSet{T <: Int, TF <: AbstractFloat, S <: Int, GInt <: Int}
#   # structure for keeping track of particle weights and states
#
#   n_particles::T  # number of particles
#   log_w::Vector{TF}  # unnormalized log-importance weights for SMC
#   log_p::Vector{TF}  # log-prob from prior
#   unique_state_id::Vector{S}  # pointers to corresponding state in `uniqueParticleStates` vector
#   proposals::Vector{Tuple{GInt,GInt}} # vector of edge tuples for current proposals
#   # proposals_map::Vector{Tuple{GInt,GInt}} # proposals mapped to vertex set of `G_data`
#   ess::String # type of effective sample size to use
#   # immortal_particles::Vector{T} # vector of particle indices to condition on surviving resampling (typically `[1]`)
#
# end


function rw_smc!(particle_container::Array{Array{ParticleState,1},1},n_particles::Int64,s_state::SamplerState)

  T = s_state.ne_data[1]
  nv_max = s_state.nv_data[1]
  # n_particles = length(particle_container)

  # initialize/pre-allocate particle paths
  # particle_container = Array{Array{ParticleState,1},1}(n_particles)
  # for p in 1:n_particles
  #   particle_container[p] = [initialize_blank_particle_state(t,T,nv_max) for t in 1:T]
  # end

  # reset particle_container
  # ***** might be unnecessary *****
  reset_particles!(particle_container)

  # sample t=1
  t = 1
  edge_idx = sample(1:T,n_particles)
  log_weights = -log(n_particles)
  updateParticles!(particle_container,t,s_state,edge_idx,[zero(Int64)],-log(T)*ones(Float64,n_particles))

  ancestors = zeros(Int64,T,n_particles)
  for t in 2:(T-2)

    # generate exhaustive proposal set
    n_edges_in_queue = zeros(Int64,n_particles)
    for p = 1:n_particles
      n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
    end
    cumu_eq = cumsum(n_edges_in_queue)
    total_eq = cumu_eq[end]
    println(total_eq)

    edge_proposals = zeros(Int64,total_eq)
    edge_logp = zeros(Float64,total_eq)
    for p = 1:n_particles
      p==1 ? (idx = 1:cumu_eq[p]) : (idx = (cumu_eq[p-1]+1):cumu_eq[p])
      edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
      edge_logp[idx] = generateProposalProbsRW(particle_container[p][t-1],s_state)
    end

    # sample edges for propogation
    edge_samples_idx = stratifiedResample(logSumExpWeights(edge_logp),n_particles) # index of edge in s_state.data_elist
    edge_samples = edge_proposals[edge_samples_idx]

    # keep track of ancestors
    ancestors[t,:] = getAncestors(edge_samples_idx,cumu_eq)

    # update particles
    updateParticles!(particle_container,t,s_state,edge_samples,ancestors[t,:],edge_logp)

  end

  t = T-1
  # for step T-1, need to look ahead to last step for weights
  # generate exhaustive proposal set
  n_edges_in_queue = zeros(Int64,n_particles)
  for p = 1:n_particles
    n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
  end
  cumu_eq = cumsum(n_edges_in_queue)
  total_eq = cumu_eq[end]
  assert(total_eq <= 2*n_particles)

  edge_proposals = zeros(Int64,total_eq)
  edge_logp = zeros(Float64,total_eq)
  last_edge_proposals = zeros(Int64,total_eq)
  for p = 1:n_particles
    p==1 ? (idx = 1:cumu_eq[p]) : (idx = (cumu_eq[p-1]+1):cumu_eq[p])
    edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
    edge_logp[idx] = generateProposalProbsRW(particle_container[p][t-1],s_state)

    # look ahead to last step for proper weighting
    for i = 1:length(idx)
      temp_edge_list = [particle_container[p][t-1].edge_list; s_state.data_elist[[edge_proposals[idx[i]]],:]]

      temp_edge_queue = particle_container[p][t-1].edge_queue
      temp_vertex_unmap = particle_container[p][t-1].vertex_unmap
      new_vertex = (particle_container[p][t-1].vertex_unmap[s_state.data_elist[[edge_proposals[idx[i]]],:]] .== 0)
      if sum(new_vertex) > 0
        vtx = s_state.data_elist[[edge_proposals[idx[i]]],find(new_vertex)]
        temp_vertex_unmap[vtx] = sum(temp_vertex_unmap .> 0) + 1
        edges_to_add = [ in(vtx,s_state.data_elist[m,:]) for m=1:T ]
        temp_edge_queue[edges_to_add] = true
      end
      temp_edge_queue[edge_proposals[idx[i]]] = false
      temp_degrees = particle_container[p][t-1].degrees
      temp_degrees[temp_vertex_unmap[s_state.data_elist[[edge_proposals[idx[i]]],:]]] += 1

      assert(sum(temp_edge_queue)==1)
      edge_logp[idx[i]] += generateProposalProbsRW(temp_edge_list,temp_vertex_unmap,t,temp_degrees,temp_edge_queue,s_state)[1]
      last_edge_proposals[idx[i]] = find(temp_edge_queue)
    end

  end

  # sample edges for propogation
  edge_samples = stratifiedResample(logSumExpWeights(edge_logp),n_particles) # index of edge in s_state.data_elist

  # keep track of ancestors
  ancestors[T-1,:] = getAncestors(edge_samples,cumu_eq)
  ancestors[T,:] = 1:n_particles # last step is deterministic

  # update particles for steps T-1 and T
  updateParticles!(particle_container,T-1,edge_samples,ancestors[T-1,:],edge_logp)
  updateParticles!(particle_container,T,edge_samples,ancestors[T,:],zeros(Float64,n_particles))

end

function getAncestors(es::Array{Int64,1},ceq::Array{Int64,1})

  anc = zeros(Int64,length(es))
  for i = 1:length(es)
    anc[i] = findfirst( ceq .>= es[i] )
  end
  return anc
end

function updateParticles!(particle_container::Array{Array{ParticleState,1},1},t::Int64,s_state::SamplerState,
            sampled_edges::Vector{Int64},ancestors::Vector{Int64},log_w::Vector{Float64})
"""
  Updates particle_container at step t to append newly sampled edges
"""
  T = length(particle_container[1])

  if t==1
    for p = 1:length(particle_container)
      particle_container[p][1].edge_list[1,:] = s_state.data_elist[sampled_edges[p],:] # add new edge
      particle_container[p][1].n_vertices[:] = 2 # update number of vertices
      particle_container[p][1].new_vertex[:] = true # update new_vertex flag
      particle_container[p][1].log_w[:] = log_w[p]
      particle_container[p][1].degrees[1:2] = 1 # update degrees
      particle_container[p][1].vertex_map[1:2] = s_state.data_elist[sampled_edges[p],:] # update vertex map
      particle_container[p][1].vertex_unmap[s_state.data_elist[sampled_edges[p],:]] = [1,2] # update vertex unmap
      # construct initial edge queue
      eq = [ ( in(s_state.data_elist[m,1],s_state.data_elist[sampled_edges[p],:]) || in(s_state.data_elist[m,2],s_state.data_elist[sampled_edges[p],:]) ) for m in 1:T ]
      eq[sampled_edges[p]] = false
      particle_container[p][1].edge_queue[:] = eq[:]
    end

  else

    for p = 1:n_particles
      # update edge list
      particle_container[p][t].edge_list[1:(t-1),:] = particle_container[ancestors[p]][t-1].edge_list[1:(t-1),:] # copy previous edges
      particle_container[p][t].edge_list[t,:] = s_state.data_elist[sampled_edges[p],:] # add new edge

      # update vertex_map
      particle_container[p][t].vertex_map[:] = particle_container[ancestors[p]][t-1].vertex_map[:] # copy previous vertex_map
      new_vertex = [ particle_container[ancestors[p]][t-1].vertex_unmap[s_state.data_elist[sampled_edges[p],1]]==0;
                      particle_container[ancestors[p]][t-1].vertex_unmap[s_state.data_elist[sampled_edges[p],2]]==0 ]
      # new_vertex = [!in(s_state.data_elist[sampled_edges[p],1],particle_container[p][t].vertex_map);
      #                 !in(s_state.data_elist[sampled_edges[p],2],particle_container[p][t].vertex_map) )]
      assert(0 <= sum(new_vertex) <= 1)
      particle_container[p][t].new_vertex[:] = (new_vertex[1] || new_vertex[2])
      particle_container[p][t].n_vertices[:] = sum(particle_container[p][t].vertex_map .> 0) + sum(new_vertex)
      sum(new_vertex)==1 ? particle_container[p][t].vertex_map[particle_container[p][t].n_vertices] = s_state.data_elist[sampled_edges[p],find(new_vertex)] : nothing

      # update log_w
      particle_container[p][t].log_w[:] = log_w[p]

      # update ancestor
      particle_container[p][t].ancestor[:] = ancestors[p]

      # update vertex_unmap
      particle_container[p][t].vertex_unmap[:] = particle_container[ancestors[p]][t-1].vertex_unmap[:]
      any(new_vertex) ? particle_container[p][t].vertex_unmap[particle_container[p][t].edge_list[t,find(new_vertex)]] = particle_container[p][t].n_vertices : nothing

      # update degrees
      particle_container[p][t].degrees[:] = particle_container[ancestors[p]][t-1].degrees[:]
      particle_container[p][t].degrees[particle_container[p][t].vertex_unmap[particle_container[p][t].edge_list[t,:]]] += 1

      # update edge queue
      particle_container[p][t].edge_queue[:] = particle_container[ancestors[p]][t-1].edge_queue[:]
      if any(new_vertex) # add new edges to the queue
        # mark edges
        vtx = s_state.data_elist[sampled_edges[p],find(new_vertex)]
        edges_to_add = [ in(vtx,s_state.data_elist[m,:]) for m=1:T ]
        particle_container[p][t].edge_queue[edges_to_add] = true

      end
      particle_container[p][t].edge_queue[sampled_edges[p]] = false # remove new edge from queue

    end

  end

end



function generateProposalProbsRW(p_state::ParticleState,s_state::SamplerState)
  """
    Returns the log probability under the prior of the entire set of eligible edges,
    as indexed in the enumeration of data edges.
  """
  return generateProposalProbsRW(p_state.edge_list,p_state.vertex_unmap,p_state.n_edges[1],p_state.degrees[1:p_state.n_vertices[1]],p_state.edge_queue,s_state)
end


function generateProposalProbsRW(edge_list::Array{Int64,2},vertex_unmap::Vector{Int64},n_edges::Int64,degrees::Vector{Int64},edge_queue::Vector{Bool},s_state::SamplerState)
"""
  Returns the log probability under the prior of the entire set of eligible edges,
  as indexed in the enumeration of data edges.
"""

  t_step = n_edges + 1
  t_remaining = s_state.ne_data[1] - n_edges
  # calculate predictive probabiliy of a new vertex
  P_I_1 = (s_state.a_α[1] + sum(s_state.I) - s_state.I[t_step])/(s_state.a_α[1] + s_state.b_α[1] + s_state.ne_data[1] - 1);
  # calculate parameters of predictive distribution for random walk length
  a_lambdaPrime = (s_state.a_λ[1] + sum(s_state.K) - s_state.K[t_step]);
  b_lambdaPrime = 1/(s_state.b_λ[1] + s_state.ne_data[1]);

  # eigenvalue decomposition
  L = normalizedLaplacian(vertex_unmap[edge_list])
  esys = eigfact(Array(Symmetric(L)))
  esys[:values][1] = zero(Float64) # for numerical stability
  esys[:values][end] = min(esys[:values][end],2.0) # for numerical stability

  # calculate r.w. probs (negative binomial predictive distribution)
  eig_pgf = ((1 - b_lambdaPrime)^(a_lambdaPrime)).*( (1 - esys[:values]).*((1 - b_lambdaPrime.*(1 - esys[:values])).^(-a_lambdaPrime)) );
  any(isinf.(eig_pgf)) ? eig_pgf[ isinf.(eig_pgf) ] = zero(eltype(eig_pgf)) : nothing # moore-penrose pseudoinverse
  W = sparse(Diagonal(degrees.^(-1/2))) * esys[:vectors] * sparse(Diagonal(eig_pgf)) * esys[:vectors]' * sparse(Diagonal(degrees.^(1/2)))
  # rwProbs = diag(degrees.^(-1/2)) * U * diag(eigenValues) * U' * diag(degrees.^(1/2));

  # account for initial vertex choice
  if s_state.size_bias[1] # size-biased selection
    for m = 1:size(W,1)
      s_state.size_bias[1] ? W[m,:] = W[m,:].*degrees[m]./(2*n_edges) : nothing
    end
  else # uniform selection
    W = W./size(W,1)
  end
  elMax!(W, 0.0) # for numerical stability
  elMin!(W, 1.0) # for numerical stability
  W = W./sum(W) # for numerical stability

  edge_proposal_idx = find(edge_queue)
  n_eligible = sum(edge_queue)
  log_p = zeros(Float64,n_eligible)

  adj = edgelist2adj(vertex_unmap[edge_list])
  # log_q = zeros(Float64,n_eligible)
  # edge_proposal_idx = find(p_state.eligible)
  for n = 1:n_eligible
    ed = [ vertex_unmap[s_state.data_elist[edge_proposal_idx[n],1]]; vertex_unmap[s_state.data_elist[edge_proposal_idx[n],2]] ]
    if ed[1] > 0 && ed[2] > 0 # vertices not yet inserted have index 0 in current particle state
      log_p[n] = log(1 - P_I_1) + log(W[ed[1],ed[2]] + W[ed[2],ed[1]])
    else
      root_vtx = find([ed[1],ed[2]])[1]
      log_p[n] = log(P_I_1 + (1 - P_I_1)*(W[ed[root_vtx],ed[root_vtx]] + sum(W[ed[root_vtx],adj[ed[root_vtx],:].==1]) )) - log(s_state.nv_data[1] - size(W,1)) # last term accounts for the "random permutation" applied to obtain the observed labels
    end
  end

  return log_p
  # return log(sum(exp.(log_p))).*ones(Float64,n_eligible) # return importance weight
end

function marginalLogLikelihodEstimate(particle_container::Array{Array{ParticleState,1},1},T::Int64)
"""
  Returns log of (unbiased) marginal likelihood estimate up to and including
  step T of input particle set
"""
  n_particles = length(particle_container)
  log_p = -T*log(n_particles)
  for t = 1:T
    tmp_w = zero(Float64)
    for p = 1:n_particles
      tmp_w += exp(particle_container[p][t].log_w)
    end
    log_p += log(tmp_w)
  end
  return log_p
end
