# functions for random walk SMC
using StatsBase

# include("Utils.jl")

# abstract type EdgeList <: Array{Int64,2} end # stores edges in a graph
# abstract type VertexMap <: Vector end
# abstract type ParticlePath <: Vector end # stores retained smc particle path

struct ParticleState
  # structure for keeping track of state information for particle states

  n_edges::Vector{Int64} # how many edges have been inserted
  edge_list::Array{Int64,2} # edges in order of insertion (pre-allocate)
  edge_idx_list::Array{Int64,1} # same as above, but indexes into data edge_list
  nbd_list::Array{Int64,2} # vector of sparse vectors, each of which stores vertex neighborhoods
  n_vertices::Vector{Int64} # number of edges in current graph
  new_vertex::Vector{Bool} # boolean indicating whether the most recent edge inserted a new vertex
  log_w::Vector{Float64} # unnormalized log-importance weight corresponding to edge_list proposals
  ancestor::Vector{Int64} # index of ancestor particle
  degrees::Vector{Float64} # degree vector
  vertex_map::Vector{Int64}  # vector where `v[j]` maps vertex `j` in the current graph to `v[j]` of the observed (final) graph
  vertex_unmap::Vector{Int64} # vector where v[j] maps vertex j in the final graph to v[j] in the current graph
  edge_queue::Vector{Bool} # vector of categories indicating which edges in the final graph are currently eligible to be inserted (=true)
  has_eigensystem::Vector{Bool} # indicates whether eigenvalue decomposition has been computed yet
  eig_vals::Vector{Float64} # eigenvalues of normalized Laplacian
  eig_vecs::Array{Float64,2} # eigenvectors of normalized Laplacian (transposed to optimize computations so that rows are eigenvectors)
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
  B::Vector{Int64} # current vector of coin flips, updates with sampler
  K::Vector{Int64} # current vector of r.w. lengths, updates with sampler
  k_range::Vector{Int64} # where to truncate support of conditional distribution of K
  b_range::Vector{Int64} # {0,1}
  particle_path::Vector{Int64} # index of current graph sequence in particle_container

end

function new_sampler_state(elist_data::Array{Int64,2},
                            size_bias::Bool,
                            a_α::Float64,b_α::Float64,
                            a_λ::Float64,b_λ::Float64,
                            α::Float64,λ::Float64,
                            B::Vector{Int64},K::Vector{Int64},k_trunc::Int64,
                            particle_path::Array{Int64,1})
"""
  Construct new sampler state object from given data and parameters
"""
  BB = copy(B)
  KK = copy(K)
  BB[1] == zero(Int64) ? nothing : BB[1] = zero(Int64)
  KK[1] == zero(Int64) ? nothing : KK[1] = zero(Int64)
  pp = copy(particle_path)
  return SamplerState(elist_data, # edge list
                      getDegrees(elist_data), # degrees
                      [size(elist_data,1)], # number of edges
                      [maximum(elist_data)], # number of vertices
                      [size_bias],
                      [a_α],[b_α],[a_λ],[b_λ], # hyper-parameters
                      [α],[λ], # parameters
                      BB, # B vector
                      KK, # K vector
                      collect(0:(k_trunc)), # k_range
                      [zero(Int64),one(Int64)], # b_range
                      pp) # particle_path

end


function initialize_blank_particle_state(n_edges::Int64,n_edges_max::Int64,deg_max::Int64,n_vertices_max::Int64)

  ps = ParticleState([n_edges],
                      zeros(Int64,n_edges,2), # edge_list
                      zeros(Int64,n_edges), # edge_idx_list
                      zeros(Int64,deg_max,n_vertices_max), # nbd_list
                      zeros(Int64,1), # n_vertices
                      [false], # new_vertex
                      zeros(Float64,1), # log_w
                      zeros(Int64,1), # ancestor
                      zeros(Int64,n_vertices_max), # degrees
                      zeros(Int64,n_vertices_max), # vertex_map
                      zeros(Int64,n_vertices_max), # vertex_unmap
                      falses(n_edges_max), # edge_queue
                      zeros(Bool,1), # has_eigensystem
                      zeros(Float64,n_vertices_max), # eig_vals
                      zeros(Float64,n_vertices_max,n_vertices_max) # eig_vecs
                      )
                      # [spzeros(deg_max) for i=1:n_vertices_max], # nbd_list
                      # spzeros(Float64,n_vertices_max), # eig_vals
                      # spzeros(Float64,n_vertices_max,n_vertices_max) # eig_vecs
  return ps
end

function reset_particle_state!(particle::ParticleState)
"""
  Function to reset the fields of a particle state
"""
  # leave .n_edges unchanged
  particle.edge_list[:] = zero(Int64)
  particle.edge_idx_list[:] = zero(Int64)
  # for i = 1:length(particle.nbd_list)
  #   particle.nbd_list[i][:] = zero(Int64)
  #   dropzeros!(particle.nbd_list[i])
  # end
  particle.nbd_list[:] = zero(Int64)
  particle.n_vertices[:] = zero(Int64)
  particle.new_vertex[:] = false
  particle.log_w[:] = zero(Int64)
  particle.ancestor[:] = zero(Int64)
  particle.degrees[:] = zero(Int64)
  particle.vertex_map[:] = zero(Int64)
  particle.vertex_unmap[:] = zero(Int64)
  particle.edge_queue[:] = false
  particle.has_eigensystem[:] = false
  particle.eig_vals[:] = zero(Float64)
  # dropzeros!(particle.eig_vals)
  particle.eig_vecs[:] = zero(Float64)
  # dropzeros!(particle.eig_vecs)
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

function reset_particles!(particle_container::Array{Array{ParticleState,1},1},particle_path::Vector{Int64})

  np = length(particle_container)::Int64
  ne = length(particle_container[1])::Int64

  for p = 1:np
    for t = 1:ne
      p == particle_path[t] ? nothing : reset_particle_state!(particle_container[p][t])
    end
  end

end

function copyParticlelState!(particle_target::ParticleState,particle_source::ParticleState)

  particle_target.n_edges[:] = particle_source.n_edges[:]
  particle_target.edge_list[:] = particle_source.edge_list[:]
  particle_target.edge_idx_list[:] = particle_source.edge_idx_list[:]
  particle_target.nbd_list[:] = particle_source.nbd_list[:]
  particle_target.n_vertices[:] = particle_source.n_vertices[:]
  particle_target.new_vertex[:] = particle_source.new_vertex[:]
  particle_target.log_w[:] = particle_source.log_w[:]
  particle_target.ancestor[:] = particle_source.ancestor[:]
  particle_target.degrees[:] = particle_source.degrees[:]
  particle_target.vertex_map[:] = particle_source.vertex_map[:]
  particle_target.vertex_unmap[:] = particle_source.vertex_unmap[:]
  particle_target.edge_queue[:] = particle_source.edge_queue[:]
  particle_target.has_eigensystem[:] = particle_source.has_eigensystem[:]
  particle_target.eig_vals[:] = particle_source.eig_vals[:]
  particle_target.eig_vecs[:] = particle_source.eig_vecs[:]

end

# function copyParticlelState!(particle_target::ParticleState,max_idx::Int64,particle_source::ParticleState)
#
#   particle_target.n_edges .= particle_source.n_edges
#   particle_target.edge_list[1:max_idx] .= particle_source.edge_list
#   particle_target.n_vertices .= particle_source.n_vertices
#   particle_target.new_vertex .= particle_source.new_vertex
#   particle_target.log_w .= particle_source.log_w
#   particle_target.ancestor .= particle_source.ancestor
#   particle_target.degrees .= particle_source.degrees
#   particle_target.vertex_map .= particle_source.vertex_map
#   particle_target.vertex_unmap .= particle_source.vertex_unmap
#   particle_target.edge_queue .= particle_source.edge_queue
#
# end

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

function rw_csmc!(particle_container::Array{Array{ParticleState,1},1},
                  s_state::SamplerState,
                  n_particles::Int64,
                  L::Array{Float64,2},
                  W::Array{Float64,2},
                  eig_pgf::Array{Float64,1},
                  ancestors::Array{Int64,2},
                  ancestors_ed::Array{Int64,2},
                  n_edges_in_queue::Array{Int64,1},
                  edge_samples_idx::Array{Int64,1},
                  edge_samples::Array{Int64,1},
                  log_w::Array{Float64,1},
                  free_edge_samples_idx::Array{Int64,1},
                  free_edge_samples::Array{Int64,1}
                )

  T = s_state.ne_data[1]::Int64
  nv_max = s_state.nv_data[1]::Int64

  # reset particle_container except for conditioned path
  # ***** might be unnecessary *****
  reset_particles!(particle_container,s_state.particle_path)
  resetArray!(L)
  resetArray!(W)
  resetArray!(eig_pgf)

  resetArray!(ancestors)
  resetArray!(ancestors_ed)
  resetArray!(n_edges_in_queue)
  resetArray!(edge_samples_idx)
  resetArray!(edge_samples)
  resetArray!(log_w)
  resetArray!(free_edge_samples_idx)
  resetArray!(free_edge_samples)

    # pre-allocated arrays
  # ancestors = zeros(Int64,T,n_particles)
  # ancestors_ed = zeros(Int64,T,n_particles)
  #
  # n_edges_in_queue = zeros(Int64,n_particles)
  # edge_samples_idx = zeros(Int64,n_particles)
  # edge_samples = zeros(Int64,n_particles)
  # log_w = zeros(Float64,n_particles)
  # free_edge_samples_idx = zeros(Int64,n_particles-1)
  # free_edge_samples = zeros(Int64,n_particles-1)

  # sample t=1
  t = one(Int64)
  # freeParticles!(free_particles,t,T,s_state.particle_path)
  edge_idx = sample(1:T,n_particles-1)::Array{Int64,1}
  fillParticles!(edge_samples,edge_idx,t,s_state.particle_path,particle_container)

  log_weights = -log(n_particles)::Float64
  updateParticles!(particle_container,t,s_state,edge_samples,zeros(Int64,1,1),-log(T)*ones(Float64,n_particles))
  ancestors[1,:] .= 1:n_particles
  ancestors_ed[1,1:(s_state.particle_path[t]-1)] .= edge_idx[1:(s_state.particle_path[t]-1)]
  ancestors_ed[1,s_state.particle_path[t]] = particle_container[s_state.particle_path[t]][t].edge_idx_list[t]
  ancestors_ed[1,(s_state.particle_path[t]+1):end] .= edge_idx[s_state.particle_path[t]:end]

  for t = 2:(T-2)

    # generate exhaustive proposal set
    for p = 1:n_particles
      n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
    end
    cumu_eq = cumsum(n_edges_in_queue)::Array{Int64,1}
    total_eq = cumu_eq[end]::Int64
    edge_logp = zeros(Float64,total_eq)
    lse_w = zeros(Float64,total_eq)
    edge_proposals = zeros(Int64,total_eq)

    # freeParticles!(free_particles,t,n_particles,s_state.particle_path)
    for p = 1:n_particles
      p==1 ? (idx = 1:cumu_eq[p]) : (idx = (cumu_eq[p-1]+1):cumu_eq[p])
      dup = findfirst( (ancestors_ed[t-1,p] .== ancestors_ed[t-1,:]) )::Int64
      if dup==p
        edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
        edge_logp[idx] = generateProposalProbsRW!(L,W,eig_pgf,particle_container[p][t-1].n_vertices[1],particle_container[p][t-1],s_state)
        resetArray!(L)
        resetArray!(W)
        resetArray!(eig_pgf)
      else # same ancestor edge
        dup==1 ? (dup_idx = 1:cumu_eq[dup]) : (dup_idx = (cumu_eq[dup-1]+1):cumu_eq[dup])
        # println(string(p) * " idx " * string(idx) * ", " * string(dup) * " idx " * string(dup_idx))
        edge_proposals[idx] .= edge_proposals[dup_idx]
        edge_logp[idx] .= edge_logp[dup_idx]
      end
    end

    # sample edges for propogation
    logSumExpWeights!(lse_w,edge_logp)
    stratifiedResample!(free_edge_samples_idx,lse_w,n_particles-1) # index of edge in s_state.data_elist
    fillParticleIdx!(edge_samples_idx,free_edge_samples_idx,t,edge_proposals,cumu_eq,s_state.particle_path,particle_container) # fill in edge sample indices

    # free_edge_samples[:] = edge_proposals[free_edge_samples_idx]
    # fillParticles!(edge_samples,free_edge_samples,t,s_state.particle_path,particle_container) # fill in edge samples
    edge_samples[:] = edge_proposals[edge_samples_idx]
    log_w[:] = edge_logp[edge_samples_idx]

    # keep track of ancestors
    getAncestors!(ancestors,ancestors_ed,t,edge_samples_idx,cumu_eq)
    # println( "step " * string(t) * ", assigned ancestor is " * string(ancestors[t,s_state.particle_path[t]]) *
            # ", path ancestor is " * string(s_state.particle_path[t-1]) )
    assert( ancestors[t,s_state.particle_path[t]]==s_state.particle_path[t-1] )

    # update particles
    updateParticles!(particle_container,t,s_state,edge_samples,ancestors,log_w)

  end

  t = (T-1)::Int64
  # for step T-1, need to look ahead to last step for weights
  # generate exhaustive proposal set
  for p = 1:n_particles
    n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
  end
  cumu_eq = cumsum(n_edges_in_queue)
  total_eq = cumu_eq[end]
  assert(total_eq <= 2*n_particles)
  edge_logp = zeros(Float64,total_eq)
  # log_w = zeros(Float64,total_eq)
  lse_w = zeros(Float64,total_eq)
  edge_proposals = zeros(Int64,total_eq)

  last_edge_proposals = zeros(Int64,total_eq)
  new_vertex = zeros(Bool,2)
  temp_edge_list = zeros(Int64,t,2)
  # deg_max = convert(Int64,maximum(s_state.degrees))::Int64
  tmp_nbd_list = Array{Int64,1}[]
  temp_edge_queue = zeros(Bool,T)
  temp_vertex_unmap = zeros(Int64,nv_max)
  edges_to_add = zeros(Bool,T)
  temp_degrees = zeros(Float64,nv_max)
  temp_nv = zeros(Int64,1)

  for p = 1:n_particles
    p==1 ? (idx = 1:cumu_eq[p]) : (idx = (cumu_eq[p-1]+1):cumu_eq[p])
    dup = findfirst( (ancestors_ed[t-1,p] .== ancestors_ed[t-1,:]) )::Int64
    if dup==p
      edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
      edge_logp[idx] = generateProposalProbsRW!(L,W,eig_pgf,particle_container[p][t-1].n_vertices[1],particle_container[p][t-1],s_state)
      resetArray!(L)
      resetArray!(W)
      resetArray!(eig_pgf)
    else # same ancestor edge
      dup==1 ? (dup_idx = 1:cumu_eq[dup]) : (dup_idx = (cumu_eq[dup-1]+1):cumu_eq[dup])
      edge_proposals[idx] .= edge_proposals[dup_idx]
      edge_logp[idx] .= edge_logp[dup_idx]
    end
    # look ahead to last step for proper weighting
    for i = 1:length(idx)
      temp_edge_list[1:(t-1),:] .= particle_container[p][t-1].edge_list
      temp_edge_list[t,:] .= s_state.data_elist[edge_proposals[idx[i]],:]

      temp_edge_queue .= particle_container[p][t-1].edge_queue
      temp_vertex_unmap .= particle_container[p][t-1].vertex_unmap
      new_vertex[:] = (particle_container[p][t-1].vertex_unmap[s_state.data_elist[[edge_proposals[idx[i]]],:]] .== 0)
      temp_nv[:] = particle_container[p][t-1].n_vertices[:] + sum(new_vertex)
      if sum(new_vertex) > 0
        vtx = s_state.data_elist[[edge_proposals[idx[i]]],find(new_vertex)][1]::Int64
        temp_vertex_unmap[vtx] = sum(temp_vertex_unmap .> 0) + 1
        edges_to_add[:] = [ in(vtx,s_state.data_elist[m,:]) for m=1:T ]
        temp_edge_queue[edges_to_add] = true
      end
      temp_edge_queue[edge_proposals[idx[i]]] = false
      temp_degrees .= particle_container[p][t-1].degrees
      temp_degrees[temp_vertex_unmap[s_state.data_elist[[edge_proposals[idx[i]]],:]]] += 1.0

      # sum(temp_edge_queue)==1 ? nothing : error("uh oh, " * sum(temp_edge_queue))
      assert(sum(temp_edge_queue)==1)
      edge_logp[idx[i]] += generateProposalProbsRW!(L,W,eig_pgf,temp_nv[1],temp_edge_list,tmp_nbd_list,temp_vertex_unmap,t,temp_degrees,temp_edge_queue,s_state)[1]
      resetArray!(L)
      resetArray!(W)
      eig_pgf[:] = zero(Float64)
      last_edge_proposals[idx[i]] = find(temp_edge_queue)[1]
    end

  end

    # sample edges for propogation
    logSumExpWeights!(lse_w,edge_logp)
    stratifiedResample!(free_edge_samples_idx,lse_w,n_particles-1) # index of edge in s_state.data_elist
    fillParticleIdx!(edge_samples_idx,free_edge_samples_idx,t,edge_proposals,cumu_eq,s_state.particle_path,particle_container) # fill in edge sample indices
    # free_edge_samples[:] = edge_proposals[free_edge_samples_idx]
    # fillParticles!(edge_samples,free_edge_samples,t,s_state.particle_path,particle_container) # fill in edge samples
    edge_samples[:] = edge_proposals[edge_samples_idx]

    log_w[:] = edge_logp[edge_samples_idx]

    # keep track of ancestors
    getAncestors!(ancestors,T-1,edge_samples_idx,cumu_eq)
    ancestors[T,:] = 1:n_particles # last step is deterministic

    # update particles for steps T-1 and T
    updateParticles!(particle_container,T-1,s_state,edge_samples,ancestors,log_w)
    edge_samples[:] = last_edge_proposals[edge_samples_idx]
    updateParticles!(particle_container,T,s_state,edge_samples,ancestors,zeros(Float64,n_particles))

end


function rw_smc!(particle_container::Array{Array{ParticleState,1},1},n_particles::Int64,s_state::SamplerState)

  T = s_state.ne_data[1]::Int64
  nv_max = s_state.nv_data[1]::Int64

  # reset particle_container
  # ***** might be unnecessary *****
  reset_particles!(particle_container)

  # pre-allocate arrays for edge probability computation
  L = zeros(Float64,nv_max,nv_max)
  W = zeros(Float64,nv_max,nv_max)
  eig_pgf = zeros(Float64,nv_max)

  # sample t=1
  t = one(Int64)
  edge_idx = sample(1:T,n_particles)::Array{Int64,1}
  log_weights = -log(n_particles)::Float64
  updateParticles!(particle_container,t,s_state,edge_idx,zeros(Int64,1,1),-log(T)*ones(Float64,n_particles))

  ancestors = zeros(Int64,T,n_particles)
  ancestors[1,:] .= 1:n_particles
  ancestors_ed = zeros(Int64,T,n_particles)
  ancestors_ed[1,:] .= edge_idx
  # pre-allocated arrays
  n_edges_in_queue = zeros(Int64,n_particles)
  edge_samples_idx = zeros(Int64,n_particles)
  edge_samples = zeros(Int64,n_particles)
  log_w = zeros(Float64,n_particles)

  for t in 2:(T-2)

    # generate exhaustive proposal set
    for p = 1:n_particles
      n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
    end
    cumu_eq = cumsum(n_edges_in_queue)::Array{Int64,1}
    total_eq = cumu_eq[end]::Int64
    edge_logp = zeros(Float64,total_eq)
    lse_w = zeros(Float64,total_eq)
    edge_proposals = zeros(Int64,total_eq)
    # println(total_eq)

    for p = 1:n_particles
      p==1 ? (idx = 1:cumu_eq[p]) : (idx = (cumu_eq[p-1]+1):cumu_eq[p])
      dup = findfirst( (ancestors_ed[t-1,p] .== ancestors_ed[t-1,:]) )::Int64
      if dup==p
        edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
        edge_logp[idx] = generateProposalProbsRW!(L,W,eig_pgf,particle_container[p][t-1].n_vertices[1],particle_container[p][t-1],s_state)
        resetArray!(L)
        resetArray!(W)
        resetArray!(eig_pgf)
      else # same ancestor edge
        dup==1 ? (dup_idx = 1:cumu_eq[dup]) : (dup_idx = (cumu_eq[dup-1]+1):cumu_eq[dup])
        edge_proposals[idx] .= edge_proposals[dup_idx]
        edge_logp[idx] .= edge_logp[dup_idx]
      end
    end

    # sample edges for propogation
    logSumExpWeights!(lse_w,edge_logp)
    stratifiedResample!(edge_samples_idx,lse_w,n_particles) # index of edge in s_state.data_elist
    edge_samples[:] = edge_proposals[edge_samples_idx]

    log_w[:] = edge_logp[edge_samples_idx]

    # keep track of ancestors
    # ancestors[t,:] = getAncestors(edge_samples_idx,cumu_eq)
    getAncestors!(ancestors,ancestors_ed,t,edge_samples_idx,cumu_eq)

    # update particles
    updateParticles!(particle_container,t,s_state,edge_samples,ancestors,log_w)

  end

  t = (T-1)::Int64
  # for step T-1, need to look ahead to last step for weights
  # generate exhaustive proposal set
  # n_edges_in_queue = zeros(Int64,n_particles)
  for p = 1:n_particles
    n_edges_in_queue[p] = sum(particle_container[p][t-1].edge_queue)
  end
  cumu_eq = cumsum(n_edges_in_queue)
  total_eq = cumu_eq[end]
  assert(total_eq <= 2*n_particles)
  edge_logp = zeros(Float64,total_eq)
  # log_w = zeros(Float64,total_eq)
  lse_w = zeros(Float64,total_eq)
  edge_proposals = zeros(Int64,total_eq)

  last_edge_proposals = zeros(Int64,total_eq)
  new_vertex = zeros(Bool,2)
  temp_edge_list = zeros(Int64,t,2)
  # deg_max = convert(Int64,maximum(s_state.degrees))::Int64
  tmp_nbd_list = Array{Int64,1}[]
  temp_edge_queue = zeros(Bool,T)
  temp_vertex_unmap = zeros(Int64,nv_max)
  edges_to_add = zeros(Bool,T)
  temp_degrees = zeros(Float64,nv_max)
  temp_nv = zeros(Int64,1)

  for p = 1:n_particles
    p==1 ? (idx = 1:cumu_eq[p]) : (idx = (cumu_eq[p-1]+1):cumu_eq[p])
    edge_proposals[idx] = find(particle_container[p][t-1].edge_queue)
    edge_logp[idx] = generateProposalProbsRW!(L,W,eig_pgf,particle_container[p][t-1].n_vertices[1],particle_container[p][t-1],s_state)
    resetArray!(L)
    resetArray!(W)
    eig_pgf[:] = zero(Float64)
    # look ahead to last step for proper weighting
    for i = 1:length(idx)
      temp_edge_list[1:(t-1),:] .= particle_container[p][t-1].edge_list
      temp_edge_list[t,:] .= s_state.data_elist[edge_proposals[idx[i]],:]

      temp_edge_queue .= particle_container[p][t-1].edge_queue
      temp_vertex_unmap .= particle_container[p][t-1].vertex_unmap
      new_vertex[:] = (particle_container[p][t-1].vertex_unmap[s_state.data_elist[[edge_proposals[idx[i]]],:]] .== 0)
      temp_nv[:] = particle_container[p][t-1].n_vertices[:] + sum(new_vertex)
      if sum(new_vertex) > 0
        vtx = s_state.data_elist[[edge_proposals[idx[i]]],find(new_vertex)][1]::Int64
        temp_vertex_unmap[vtx] = sum(temp_vertex_unmap .> 0) + 1
        edges_to_add[:] = [ in(vtx,s_state.data_elist[m,:]) for m=1:T ]
        temp_edge_queue[edges_to_add] = true
      end
      temp_edge_queue[edge_proposals[idx[i]]] = false
      temp_degrees .= particle_container[p][t-1].degrees
      temp_degrees[temp_vertex_unmap[s_state.data_elist[[edge_proposals[idx[i]]],:]]] += 1.0

      # sum(temp_edge_queue)==1 ? nothing : error("uh oh, " * sum(temp_edge_queue))
      assert(sum(temp_edge_queue)==1)
      edge_logp[idx[i]] += generateProposalProbsRW!(L,W,eig_pgf,temp_nv[1],temp_edge_list,tmp_nbd_list,temp_vertex_unmap,t,temp_degrees,temp_edge_queue,s_state)[1]
      resetArray!(L)
      resetArray!(W)
      eig_pgf[:] = zero(Float64)
      last_edge_proposals[idx[i]] = find(temp_edge_queue)[1]
    end

  end

  # sample edges for propogation
  logSumExpWeights!(lse_w,edge_logp)
  stratifiedResample!(edge_samples_idx,lse_w,n_particles) # index of edge in s_state.data_elist
  edge_samples[:] = edge_proposals[edge_samples_idx]

  log_w[:] = edge_logp[edge_samples_idx]

  # keep track of ancestors
  getAncestors!(ancestors,T-1,edge_samples_idx,cumu_eq)
  ancestors[T,:] = 1:n_particles # last step is deterministic

  # update particles for steps T-1 and T
  updateParticles!(particle_container,T-1,s_state,edge_samples,ancestors,log_w)
  edge_samples[:] = last_edge_proposals[edge_samples_idx]
  updateParticles!(particle_container,T,s_state,edge_samples,ancestors,zeros(Float64,n_particles))

end

function getAncestors(es::Array{Int64,1},ceq::Array{Int64,1})::Array{Int64,1}
  anc = zeros(Int64,length(es))
  for i = 1:length(es)
    anc[i] = findfirst( ceq .>= es[i] )
  end
  return anc
end

function getAncestors!(ancestors::Array{Int64,2},t::Int64,es::Array{Int64,1},ceq::Array{Int64,1})

  for i = 1:length(es)
    ancestors[t,i] = findfirst( ceq .>= es[i] )
  end

end

function getAncestors!(ancestors::Array{Int64,2},ancestors_ed::Array{Int64,2},t::Int64,es::Array{Int64,1},ceq::Array{Int64,1})

  for i = 1:length(es)
    ancestors[t,i] = findfirst( ceq .>= es[i] )
    # ancestors[t,i]==1 ? st_ed = zero(Int64) : st_ed = ceq[ancestors[t,i]-1]
    ancestors_ed[t,i] = es[i]
  end

end

function getAncestors!(ancestors::Array{Int64,2},ancestors_ed::Array{Int64,2},t::Int64,es_idx::Array{Int64,1},ceq::Array{Int64,1},
                        ed_p::Array{Int64,1},particle_path::Array{Int64,1},particle_container::Array{Array{ParticleState,1},1})

  for i = 1:(particle_path[t]-1)
    ancestors[t,i] = findfirst( ceq .>= es_idx[i])
    ancestors[t,i]==1 ? st_ed = zero(Int64) : st_ed = ceq[ancestors[t,i]-1]
    ancestors_ed[t,i] = es_idx[i] - st_ed
  end

  i = particle_path[t]::Int64
  ancestors[t,i] = particle_path[t-1]
  ancestors[t,i]==1 ? st_ed = zero(Int64) : st_ed = ceq[ancestors[t,i]-1]
  ancestors_ed[t,i] = es_idx[i] - st_ed

  for i = (particle_path[t]+1):length(es)
    ancestors[t,i] = findfirst( ceq .>= es_idx[i])
    ancestors[t,i]==1 ? st_ed = zero(Int64) : st_ed = ceq[ancestors[t,i]-1]
    ancestors_ed[t,i] = es_idx[i] - st_ed
  end

end

function updateParticles!(particle_container::Array{Array{ParticleState,1},1},t::Int64,s_state::SamplerState,
            sampled_edges::Vector{Int64},ancestors::Array{Int64,2},log_w::Vector{Float64})
"""
  Updates particle_container at step t to append newly sampled edges
"""
  T = length(particle_container[1])::Int64

  if t==1
    for p = 1:length(particle_container)
      particle_container[p][1].edge_list[1,:] = s_state.data_elist[sampled_edges[p],:] # add new edge
      particle_container[p][1].edge_idx_list[1] = sampled_edges[p]
      particle_container[p][1].nbd_list[1,1] = 2 # neighborhood of vertex 1
      particle_container[p][1].nbd_list[1,2] = 1 # neighborhood of vertex 1
      particle_container[p][1].n_vertices[:] = 2 # update number of vertices
      particle_container[p][1].new_vertex[:] = true # update new_vertex flag
      particle_container[p][1].log_w[:] = log_w[p]
      particle_container[p][1].degrees[1:2] = one(Float64) # update degrees
      particle_container[p][1].vertex_map[1:2] = s_state.data_elist[sampled_edges[p],:] # update vertex map
      particle_container[p][1].vertex_unmap[s_state.data_elist[sampled_edges[p],:]] = [1,2] # update vertex unmap
      # construct initial edge queue
      eq = [ ( in(s_state.data_elist[m,1],s_state.data_elist[sampled_edges[p],:]) || in(s_state.data_elist[m,2],s_state.data_elist[sampled_edges[p],:]) ) for m in 1:T ]::Vector{Bool}
      eq[sampled_edges[p]] = false
      particle_container[p][1].edge_queue .= eq
      particle_container[p][1].ancestor[:] = p # ancestor for book-keeping
    end

  else
    new_vertex = falses(2)
    edge = zeros(Int64,2)
    edge_unmap = zeros(Int64,2)
    edges_to_add = zeros(Bool,T)
    for p = 1:n_particles
      edge .= s_state.data_elist[sampled_edges[p],:]
      # update edge list
      particle_container[p][t].edge_list[1:(t-1),:] .= particle_container[ancestors[t,p]][t-1].edge_list # copy previous edges
      particle_container[p][t].edge_list[t,:] .= edge # add new edge

      # update edge index list
      particle_container[p][t].edge_idx_list[1:(t-1)] = particle_container[ancestors[t,p]][t-1].edge_idx_list
      particle_container[p][t].edge_idx_list[t] = sampled_edges[p]

      # update vertex_map
      particle_container[p][t].vertex_map .= particle_container[ancestors[t,p]][t-1].vertex_map # copy previous vertex_map
      new_vertex[:] = [ particle_container[ancestors[t,p]][t-1].vertex_unmap[edge[1]]==0;
                      particle_container[ancestors[t,p]][t-1].vertex_unmap[edge[2]]==0 ]
      # new_vertex = [!in(s_state.data_elist[sampled_edges[p],1],particle_container[p][t].vertex_map);
      #                 !in(s_state.data_elist[sampled_edges[p],2],particle_container[p][t].vertex_map) )]
      assert(0 <= sum(new_vertex) <= 1)
      particle_container[p][t].new_vertex[:] = (new_vertex[1] || new_vertex[2])
      particle_container[p][t].n_vertices[:] = sum(particle_container[p][t].vertex_map .> 0) + sum(new_vertex)
      any(new_vertex) ? particle_container[p][t].vertex_map[particle_container[p][t].n_vertices] = edge[find(new_vertex)] : nothing

      # update log_w
      particle_container[p][t].log_w[:] = log_w[p]

      # update ancestor
      particle_container[p][t].ancestor[:] = ancestors[t,p]

      # update vertex_unmap
      particle_container[p][t].vertex_unmap .= particle_container[ancestors[t,p]][t-1].vertex_unmap
      any(new_vertex) ? new_v = find(new_vertex)[1]::Int64 : nothing
      any(new_vertex) ? particle_container[p][t].vertex_unmap[particle_container[p][t].edge_list[t,new_v]] = particle_container[p][t].n_vertices[1] : nothing

      # update degrees
      edge_unmap .= particle_container[p][t].vertex_unmap[edge]
      particle_container[p][t].degrees .= particle_container[ancestors[t,p]][t-1].degrees
      particle_container[p][t].degrees[edge_unmap[1]] += 1
      particle_container[p][t].degrees[edge_unmap[2]] += 1

      # update neighborhoods
      particle_container[p][t].nbd_list[:] = particle_container[ancestors[t,p]][t-1].nbd_list[:]
      particle_container[p][t].nbd_list[convert(Int64,particle_container[p][t].degrees[edge_unmap[1]]),edge_unmap[1]] = edge_unmap[2]
      particle_container[p][t].nbd_list[convert(Int64,particle_container[p][t].degrees[edge_unmap[2]]),edge_unmap[2]] = edge_unmap[1]

      # update edge queue
      particle_container[p][t].edge_queue .= particle_container[ancestors[t,p]][t-1].edge_queue
      if new_vertex[1] || new_vertex[2] # add new edges to the queue
        # mark edges
        vtx = s_state.data_elist[sampled_edges[p],new_v][1]::Int64
        edges_to_add[:] = [ in(vtx,s_state.data_elist[m,:]) for m=1:T ]::Vector{Bool}
        particle_container[p][t].edge_queue[edges_to_add] = true

      end
      particle_container[p][t].edge_queue[sampled_edges[p]] = false # remove new edge from queue

    end

  end

end

function generateProposalProbsRW!(L::Array{Float64,2},W::Array{Float64,2},eig_pgf::Array{Float64,1},nv::Int64,p_state::ParticleState,s_state::SamplerState)::Array{Float64,1}

  !p_state.has_eigensystem[1] ? updateEigenSystem!(L,nv,p_state) : nothing

  return generateProposalProbsRW!(L,W,eig_pgf,p_state.eig_vals,p_state.eig_vecs,nv,p_state.edge_list,p_state.nbd_list,p_state.vertex_unmap,p_state.n_edges[1],p_state.degrees[1:p_state.n_vertices[1]],p_state.edge_queue,s_state)
end

function generateProposalProbsRW!(L::Array{Float64,2},W::Array{Float64,2},eig_pgf::Array{Float64,1},
        nv::Int64,edge_list::Array{Int64,2},nbd_list::Union{Array{Array{Int64,1},1},Array{Int64,2},Array{SparseVector{Int64,Int64},1}},vertex_unmap::Vector{Int64},
        n_edges::Int64,degrees::Vector{Float64},
        edge_queue::Vector{Bool},s_state::SamplerState)::Array{Float64,1}

    # need to generate eigensystem
    denseNormalizedLaplacian!(L,vertex_unmap[edge_list],degrees,nv)
    eig_vals,eig_vecs = generateEigenSystem!(L,nv)

    return generateProposalProbsRW!(L,W,eig_pgf,eig_vals,eig_vecs,nv,edge_list,nbd_list,vertex_unmap,n_edges,degrees,edge_queue,s_state)

end

function generateProposalProbsRW!(L::Array{Float64},W::Array{Float64,2},eig_pgf::Array{Float64,1},
        eig_vals::Union{Array{Float64,1},SparseVector{Float64,Int64}},
        eig_vecs::Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}},
        nv::Int64,edge_list::Array{Int64,2},nbd_list::Union{Array{Array{Int64,1},1},Array{Int64,2},Array{SparseVector{Int64,Int64},1}},vertex_unmap::Vector{Int64},
        n_edges::Int64,degrees::Vector{Float64},
        edge_queue::Vector{Bool},s_state::SamplerState)::Array{Float64,1}
"""
  Returns the log probability under the prior of the entire set of eligible edges,
  as indexed in the enumeration of data edges. Operates on L and W in-place.
"""

  t_step = (n_edges + 1)::Int64
  t_remaining = (s_state.ne_data[1] - n_edges)::Int64

  # calculate predictive probabiliy of a new vertex
  P_I_1 = ((s_state.a_α[1] + sum(s_state.B) - s_state.B[t_step])/(s_state.a_α[1] + s_state.b_α[1] + s_state.ne_data[1] - 1))::Float64

  # calculate parameters of predictive distribution for random walk length
  a_lambdaPrime = (s_state.a_λ[1] + sum(s_state.K) - s_state.K[t_step])::Float64
  b_lambdaPrime = (1/(s_state.b_λ[1] + s_state.ne_data[1]))::Float64

  # calculate r.w. probs (negative binomial predictive distribution)
  # eig_pgf = zeros(Float64,nv)
  nbPred!(eig_pgf,nv,a_lambdaPrime,b_lambdaPrime,eig_vals)
  # eig_pgf[:] = (((1.0 - b_lambdaPrime)^(a_lambdaPrime)).*( (1.0 .- esys[:values]).*((1.0 .- b_lambdaPrime.*(1.0 .- esys[:values])).^(-a_lambdaPrime)) ))::Array{Float64,1}
  # any(isinf.(eig_pgf)) ? eig_pgf[ isinf.(eig_pgf) ] = zero(Float64) : nothing # moore-penrose pseudoinverse
  # randomWalkProbs!(W,nv,degrees,eig_pgf,eig_vecs)
  # W[1:nv,1:nv] = Diagonal(degrees.^(-1/2)) * esys[:vectors] * Diagonal(eig_pgf) * esys[:vectors]' * Diagonal(degrees.^(1/2))
  # rwProbs = diag(degrees.^(-1/2)) * U * diag(eigenValues) * U' * diag(degrees.^(1/2));

  # account for initial vertex choice
  # if s_state.size_bias[1] # size-biased selection
  #   for m = 1:nv
  #     s_state.size_bias[1] ? W[1:nv,m] .= W[1:nv,m].*(degrees[1:nv]./(2*n_edges)) : nothing
  #   end
  # else # uniform selection
  #   W[1:nv,1:nv] .= W[1:nv,1:nv]./nv
  # end
  # elMax!(W[1:nv,1:nv], 0.0) # for numerical stability
  # elMin!(W[1:nv,1:nv], 1.0) # for numerical stability
  # W[1:nv,1:nv] .= W[1:nv,1:nv]./sum(W[1:nv,1:nv]) # for numerical stability

  # calculate log-probabilities
  edge_proposal_idx = find(edge_queue)::Array{Int64,1}
  n_eligible = sum(edge_queue)::Int64
  log_p = zeros(Float64,n_eligible)
  ed = zeros(Int64,1,2)
  p_tmp = zero(Float64)
  log_labelprob = log(s_state.nv_data[1] - nv)::Float64
  w = zeros(Float64,1,2)
  frwp = zeros(Float64,1)
  nbd = zeros(Int64,size(nbd_list,1))
  rt = zeros(Int64,1,2)

  if isempty(nbd_list)
    edgelist2adj!(L,vertex_unmap[edge_list]) # L is used for adjacency matrix
  end
  for n = 1:n_eligible
    resetArray!(nbd)
    resetArray!(ed)
    resetArray!(w)
    resetArray!(rt)
    ed[:] = vertex_unmap[s_state.data_elist[edge_proposal_idx[n],:]]
    if ed[1] > 0 && ed[2] > 0 # vertices not yet inserted have index 0 in current particle state
      randomWalkProbs!(w,ed,nv,degrees,eig_pgf,eig_vecs)
      s_state.size_bias[1] ? (w .*= (degrees[ed]./(2*n_edges))) : (w[:] *= 1/nv)
      log_p[n] = (log(1 - P_I_1) + log(sum(w)))::Float64
    else
      root_vtx = findfirst(ed .> 0)::Int64
      rt[:] = root_vtx
      # nbd[1] = root_vtx
      if(!isempty(nbd_list))
        nbd[:] = nbd_list[:,root_vtx]
        # nbd[:] = [root_vtx; nbd_list[:,root_vtx]]
      else
        nb = find( L[:,root_vtx] .> 0 ) # here, L is the adjacency matrix
        nbd[1:numel(nb)] = nb
      end
      randomWalkProbs!(w,rt,nv,degrees,eig_pgf,eig_vecs) # prob of ending where it starts
      frwp[1] = randomWalkProbs(root_vtx,nbd,nv,degrees,eig_pgf,eig_vecs) + w[1] # prob of fruitless r.w.
      s_state.size_bias[1] ? (frwp[:] *= degrees[root_vtx]/(2*n_edges)) : (frwp[:] *= 1/nv)
      # isempty(nbd_list) ? p_tmp = fruitlessRWProb(W,L,root_vtx,nv)::Float64 : p_tmp = fruitlessRWProb(W,nbd_list,root_vtx)::Float64
      log_p[n] = (log(P_I_1 + (1 - P_I_1)*frwp[1] ) - log_labelprob)::Float64 # last term accounts for the "random permutation" applied to obtain the observed labels
    end
  end

  return log_p
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
      tmp_w += exp(particle_container[p][t].log_w[1])
      # println(exp(particle_container[p][t].log_w[1]))
    end
    log_p += log(tmp_w)
  end
  return log_p
end


function getParticlePath!(particle_path::Array{Int64,1},particle_container::Array{Array{ParticleState,1},1},p_idx::Int64)

  T = length(particle_path)
  particle_path[T] = p_idx
  for t in (T-1):-1:1
    particle_path[t] = particle_container[particle_path[t+1]][t+1].ancestor[1]
  end

end

function fillParticles!(edge_samples::Array{Int64,1},free_edge_samples::Array{Int64,1},t::Int64,particle_path::Array{Int64,1},particle_container::Array{Array{ParticleState,1},1})

  for i = 1:(particle_path[t]-1)
    edge_samples[i] = free_edge_samples[i]
  end
  edge_samples[particle_path[t]] = particle_container[particle_path[t]][t].edge_idx_list[t]
  for i = (particle_path[t]+1):length(edge_samples)
    edge_samples[i] = free_edge_samples[i-1]
  end

end

function fillParticleIdx!(es_idx::Array{Int64,1},free_es_idx::Array{Int64,1}, t::Int64,edge_proposals::Array{Int64,1},cumu_eq::Array{Int64,1}, particle_path::Array{Int64,1},particle_container::Array{Array{ParticleState,1},1})

  for i = 1:(particle_path[t]-1)
    es_idx[i] = free_es_idx[i]
  end

  p_anc = particle_path[t-1]::Int64
  p_anc==1 ? st_ed = one(Int64) : st_ed = (cumu_eq[p_anc-1])::Int64
  es_idx[particle_path[t]] = st_ed + findfirst( particle_container[particle_path[t]][t].edge_idx_list[t] .== edge_proposals[(st_ed+1):cumu_eq[p_anc]])

  for i = (particle_path[t]+1):length(es_idx)
    es_idx[i] = free_es_idx[i-1]
  end

end

function freeParticles!(free_particles::Array{Int64,1},t::Int64,T::Int64,particle_path::Array{Int64,1})

  tt = zero(Int64)
  for i = 1:(particle_path[t]-1)
    free_particles[i] = i
  end
  for i = particle_path[t]:(T-1)
    free_particles[i] = i+1
  end

end
