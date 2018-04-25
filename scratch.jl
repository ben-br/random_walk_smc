# test bed
using Distributions
using Plots
gr()
# import LightGraphs
import MAT


include("Utils.jl")
include("rw_smc.jl")
include("rand_rw.jl")



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
g = randomWalkSimpleGraph(n_edges=100,alpha_prob=0.33,length_distribution=ld,sizeBias=true)

heatmap(full(edgelist2adj(g)))

######

# set data parameters
n_edges_data = 100
α = 0.25
λ = 4.0
ld = Poisson(λ)
sb = true

# sample graph
g = randomWalkSimpleGraph(n_edges=n_edges_data,alpha_prob=α,length_distribution=ld,sizeBias=sb)

# lg = LightGraphs.Graph(edgelist2adj(g))
# LightGraphs.is_connected(lg)

# set sampler parameters
n_particles = 10
a_α = 1.0
b_α = 1.0
a_λ = 0.25
b_λ = 1.0

# initialize empty particle container
nv_max = maximum(g)
particle_container = Array{Array{ParticleState,1},1}(n_particles)
for p in 1:n_particles
  particle_container[p] = [initialize_blank_particle_state(t,n_edges_data,nv_max) for t in 1:n_edges_data]
end

particle_path = [initialize_blank_particle_state(t,n_edges_data,nv_max) for t in 1:n_edges_data]
I_coins = rand(Bernoulli(α),n_edges_data)
K_walks = rand(ld,n_edges_data)

# initialize SamplerState
s_state = new_sampler_state(g,sb,a_α,b_α,a_λ,b_λ,α,λ,I_coins,K_walks,particle_path)


# run a sweep of SMC
rw_smc!(particle_container,n_particles,s_state)
