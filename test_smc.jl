
using Distributions
import LightGraphs

include("rw_smc.jl")
include("Utils.jl")
include("rand_rw.jl")
include("gibbs_updates.jl")

# set seed
srand(0)

# set data parameters
const n_edges_data = 100
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
const n_mcmc_iter = 10 # total number of iterations; includes burn-in
const n_burn = 0 # burn-in
const n_collect = 1 # collect a sample every n_collect iterations
n_print = 1 # print progress updates every n_print iteration

const n_particles = 100
const a_α = 1.0
const b_α = 1.0
const a_λ = 0.25
const b_λ = 1.0
const k_trunc = 10*lg_diam
const α_start = convert(Float64,(maximum(g)-1)/(n_edges_data-1))
const λ_start = convert(Float64,sqrt(lg_diam))

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
    rw_csmc!(particle_container,s_state,n_particles,
            L,W,eig_pgf,
            ancestors,
            ancestors_ed,
            n_edges_in_queue,
            edge_samples_idx,
            edge_samples,
            log_w,
            free_edge_samples_idx,
            free_edge_samples)

    p_idx = sample(1:n_particles,1)[1]
    getParticlePath!(s_state.particle_path,particle_container,p_idx)

    # update B, K, α, λ
    updateBandK!(s_state,particle_container,
                L,eig_pgf,lp_b,p_b,lp_k,p_k)
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

using Plots
gr()
plot(lambda_samples,legend=false)
plot(alpha_samples,legend=false)
plot(edge_sequence_samples',legend=false)

# updateBandK!(L,s_state,particle_container)
# updateAlphaAndLambda!(s_state)
#
# @time rw_csmc!(particle_container,s_state,n_particles,L,W,eig_pgf)
# getParticlePath!(s_state.particle_path,particle_container,p_idx)
# updateBandK!(L,s_state,particle_container)
# updateAlphaAndLambda!(s_state)

# using Plots
# gr()
# plot()
# for i = 1:n_particles
#     plot!(particle_container[i][n_edges_data].edge_idx_list,legend=false)
# end

# for p in 1:n_particles
#   particle_container[p] = [initialize_blank_particle_state(t,n_edges_data,deg_max,nv_max) for t in 1:n_edges_data]
# end
#
# function main(n_reps::Int64,particle_container::Array{Array{ParticleState,1},1},n_particles::Int64,s_state::SamplerState)
#
#     for i = 1:n_reps
#       rw_smc!(particle_container,n_particles,s_state)
#       println(marginalLogLikelihodEstimate(particle_container,n_edges_data))
#     end
#
# end
#
# @time main(5,particle_container,n_particles,s_state)
#
# particle_path = zeros(Int64,n_edges_data)
# getParticlePath!(particle_path,particle_container,5)

# Profile.clear()
# @profile rw_smc!(particle_container,n_particles,s_state)
# f = open("prof_bk_txt.txt","w")
# Profile.print(f)
# close(f)

# for i = 1:5
#   rw_smc!(particle_container,n_particles,s_state)
#   println(marginalLogLikelihodEstimate(particle_container,n_edges_data))
# end

# @time rw_smc!(particle_container,n_particles,s_state)
