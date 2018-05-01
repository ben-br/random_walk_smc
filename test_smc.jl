
using Distributions
import LightGraphs

include("rw_smc.jl")
include("Utils.jl")
include("rand_rw.jl")
include("gibbs_updates.jl")


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
const n_particles = 10
const a_α = 1.0
const b_α = 1.0
const a_λ = 0.25
const b_λ = 1.0
const k_trunc = 10*lg_diam

# initialize empty particle container
nv_max = maximum(g)::Int64
deg_max = convert(Int64,maximum(getDegrees(g)))
particle_container = Array{Array{ParticleState,1},1}(n_particles)
for p in 1:n_particles
  particle_container[p] = [initialize_blank_particle_state(t,n_edges_data,deg_max,nv_max) for t in 1:n_edges_data]
end

particle_path = zeros(Int64,n_edges_data)
B_coins = rand(Bernoulli(α),n_edges_data)
K_walks = rand(ld,n_edges_data)
B_coins[1] = zero(Int64)
K_walks[1] = zero(Int64)

L = zeros(Float64,nv_max,nv_max)

# initialize SamplerState
s_state = new_sampler_state(g,sb,a_α,b_α,a_λ,b_λ,α,λ,B_coins,K_walks,k_trunc,particle_path)

@time rw_smc!(particle_container,n_particles,s_state)
p_idx = sample(1:n_particles,1)[1]

getParticlePath!(s_state.particle_path,particle_container,p_idx)
updateBandK!(L,s_state,particle_container)

plot()
for i = 1:n_particles
    plot!(particle_container[i][n_edges_data].edge_idx_list,legend=false)
end

for p in 1:n_particles
  particle_container[p] = [initialize_blank_particle_state(t,n_edges_data,deg_max,nv_max) for t in 1:n_edges_data]
end

function main(n_reps::Int64,particle_container::Array{Array{ParticleState,1},1},n_particles::Int64,s_state::SamplerState)

    for i = 1:n_reps
      rw_smc!(particle_container,n_particles,s_state)
      println(marginalLogLikelihodEstimate(particle_container,n_edges_data))
    end

end

@time main(5,particle_container,n_particles,s_state)

particle_path = zeros(Int64,n_edges_data)
getParticlePath!(particle_path,particle_container,5)

# Profile.clear()
# @profile rw_smc!(particle_container,n_particles,s_state)
# f = open("prof_txt.txt","w")
# Profile.print(f)
# close(f)

# for i = 1:5
#   rw_smc!(particle_container,n_particles,s_state)
#   println(marginalLogLikelihodEstimate(particle_container,n_edges_data))
# end

# @time rw_smc!(particle_container,n_particles,s_state)
