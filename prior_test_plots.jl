
using JLD
import Iterators
using Plots
gr()

const n_mcmc_iter = 3000 # total number of iterations; includes burn-in
const n_burn = 500 # burn-in
const n_collect = 1 # collect a sample every n_collect iterations
const n_print = 10 # print progress updates every n_print iterations
const n_particles = 100

const n_edges_data = 50
const α = 0.25
const λ = 4.0

job_id = "2190469"

n_samples = convert(Int64,floor((n_mcmc_iter - n_burn)/n_collect))
n_samples_plot = 1500

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

alpha_s = zeros(Float64,n_samples,n_hp)
lambda_s = zeros(Float64,n_samples,n_hp)
tmp = zeros(Float64,n_samples)

dirname1 = "/data/ziz/not-backed-up/bloemred/random_walk_smc/results/prior_sensitivity_" * job_id * "/"
dirname2 = "/data/ziz/bloemred/random_walk_smc/plots/prior/ne_50/"

for n = 1:n_hp
  fname = "prior_" * job_id * "_" * string(n) * ".jld"
  tmp[:] = load(dirname1 * fname, "alpha_samples")
  alpha_s[:,n] = tmp
  tmp[:] = load(dirname1 * fname, "lambda_samples")
  lambda_s[:,n] = tmp
  plotname = dirname2 * "prior_" * string(n) * ".pdf"
  scatter(lambda_s[1:n_samples_plot,n],alpha_s[1:n_samples_plot,n],legend=false);
  scatter!([λ],[α],legend=false,c=:black,marker=:star4,ms=10);
  savefig(plotname)
end
