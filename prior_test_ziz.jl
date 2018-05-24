## script to test prior sensitivity

ENV["JULIA_PKGDIR"] = "/data/ziz/bloemred/.julia"

import Iterators
import LightGraphs
using JLD
using JSON
using DataStructures

include("rw_smc.jl")
include("Utils.jl")
include("rand_rw.jl")
include("gibbs_updates.jl")

# get SLURM job and task ids
job_id = convert(Int64,ENV["SLURM_ARRAY_JOB_ID"])
task_id = convert(Int64,ENV["SLURM_ARRAY_TASK_ID"])
job_name = ENV["SLURM_JOB_NAME"]

# set seed
sd = srand(0)

# set data parameters
const n_edges_data = 50
const α = 0.25
const λ = 4.0
const ld = Poisson(λ)
const sb = true

# sample graph
g = randomWalkSimpleGraph(n_edges=n_edges_data,alpha_prob=α,length_distribution=ld,sizeBias=sb)

lg = LightGraphs.Graph(edgelist2adj(g))
# LightGraphs.is_connected(lg)
lg_diam = LightGraphs.diameter(lg)::Int64

# set sampler parameters
const n_mcmc_iter = 2500 # total number of iterations; includes burn-in
const n_burn = 500 # burn-in
const n_collect = 1 # collect a sample every n_collect iterations
const n_print = 10 # print progress updates every n_print iterations

const n_particles = 100
const k_trunc = 10*lg_diam
const α_start = convert(Float64,0.5*(maximum(g)-1)/(n_edges_data-1))
const λ_start = convert(Float64,sqrt(lg_diam))

# hyperparameter settings
alpha_hp = [0.5 0.5;
            1.0 1.0;
            2.0 2.0]

lambda_hp = [1.0 0.25;
              20.0 2.0;
              0.01 0.01]

n_ahp = size(alpha_hp,1)
n_lhp = size(lambda_hp,1)
hp_idx = Iterators.product(1:n_ahp,1:n_lhp)
n_hp = length(hp_idx)

hp_idx_this = Iterators.nth(hp_idx,task_id)

# set prior parameters
a_α = alpha_hp[hp_idx_this[1],1]
b_α = alpha_hp[hp_idx_this[1],2]
a_λ = lambda_hp[hp_idx_this[2],1]
b_λ = lambda_hp[hp_idx_this[2],2]

# initialize empty particle container
nv_max = maximum(g)::Int64
deg_max = convert(Int64,maximum(getDegrees(g)))
particle_container = Array{Array{ParticleState,1},1}(n_particles)
for p in 1:n_particles
  particle_container[p] = [initialize_blank_particle_state(t,n_edges_data,deg_max,nv_max) for t in 1:n_edges_data]
end

pp = zeros(Int64,n_edges_data)
B_coins = rand(Bernoulli(α_start),n_edges_data)
K_walks = rand(Poisson(λ_start),n_edges_data)
B_coins[1] = zero(Int64)
K_walks[1] = zero(Int64)

# initialize SamplerState
s_state = new_sampler_state(g,sb,a_α,b_α,a_λ,b_λ,α_start,λ_start,B_coins,K_walks,k_trunc,pp)

# initialize sampler
rw_smc!(particle_container,n_particles,s_state)
# run some checks here
p_idx = sample(1:n_particles,1)[1]
getParticlePath!(s_state.particle_path,particle_container,p_idx)

# pre-allocate arrays for computation in rw_csmc!
L = zeros(Float64,nv_max,nv_max)
W = zeros(Float64,nv_max,nv_max)
eig_pgf = zeros(Float64,nv_max)
ancestors = zeros(Int64,n_edges_data,n_particles)
ancestors_ed = zeros(Int64,n_edges_data,n_particles)
unq = zeros(Int64,n_edges_data,n_particles)
n_edges_in_queue = zeros(Int64,n_particles)
edge_samples_idx = zeros(Int64,n_particles)
edge_samples = zeros(Int64,n_particles)
log_w = zeros(Float64,n_particles)
free_edge_samples_idx = zeros(Int64,n_particles-1)
free_edge_samples = zeros(Int64,n_particles-1)
lp_b = zeros(Float64,2)
p_b = zeros(Float64,2)
lp_k = zeros(Float64,k_trunc+1)
p_k = zeros(Float64,k_trunc+1)

# pre-allocate sample collection arrays
n_samples = convert(Int64,floor((n_mcmc_iter - n_burn)/n_collect))
alpha_samples = zeros(Float64,n_samples)
lambda_samples = zeros(Float64,n_samples)
particle_trajectory_samples = zeros(Int64,n_samples,n_edges_data)
edge_sequence_samples = zeros(Int64,n_samples,n_edges_data)
log_marginal_samples = zeros(Float64,n_samples)

n_s = zero(Int64)
t_elapsed = zero(Float64)
tic();
for s = 1:n_mcmc_iter
    # update edge sequence
    # @enter @time
    rw_csmc!(particle_container,s_state,n_particles,
            L,W,eig_pgf,
            ancestors,
            ancestors_ed,
            unq,
            n_edges_in_queue,
            edge_samples_idx,
            edge_samples,
            log_w,
            free_edge_samples_idx,
            free_edge_samples)

    p_idx = sample(1:n_particles,1)[1]
    getParticlePath!(s_state.particle_path,particle_container,p_idx)

    # update B, K, α, λ
    updateBandK!(s_state,particle_container,L,eig_pgf,lp_b,p_b,lp_k,p_k)
    updateAlphaAndLambda!(s_state)

    if s > n_burn && mod((s - n_burn),n_collect)==0
        # collect samples
        n_s += one(Int64)
        log_marginal_samples[n_s] = marginalLogLikelihodEstimate(particle_container,n_edges_data)
        saveSamples!(alpha_samples,lambda_samples,particle_trajectory_samples,edge_sequence_samples,
                    s_state,particle_container,p_idx,n_s)

    end

    if mod(s,n_print)==0
        t_elapsed += toq();
        println( "Finished with " * string(s) * " / " * string(n_mcmc_iter) * " iterations. Elapsed time: " * string(t_elapsed) )
        tic();
    end

end


# save sampler output
dirname = "/data/localhost/not-backed-up/bloemred/random_walk_smc/results/$(job_name)_$(job_id)/" 
fname = "prior_$(job_id)_$(task_id)_samples.jld"
pathname = dirname * fname
save(pathname,
      "g_data",g,
      "init_seed",sr,
      "final_sampler_state",s_state, # will have all relevant hyper parameters, etc
      "alpha_samples",alpha_samples,
      "lambda_samples",lambda_samples,
      "edge_sequence_samples",edge_sequence_samples, # can reconstruct final conditioned particle path from this and s_state.particle_path
      "log_marginal_samples",log_marginal_samples
      )
