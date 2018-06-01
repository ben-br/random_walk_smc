d = load("plots/prior/ne_50/agg_samples_50_2190469.jld")

alpha_s = d["alpha_samples"]
lambda_s = d["lambda_samples"]

α = d["alpha"]
λ = d["lambda"]

n_samples_plot = 1000
n_hp = 9
dirname2 = "plots/prior/ne_50/"

for n = 1:n_hp
  # fname = "prior_" * job_id * "_" * string(n) * ".jld"
  # tmp[:] = load(dirname1 * fname, "alpha_samples")
  # alpha_s[:,n] = tmp
  # tmp[:] = load(dirname1 * fname, "lambda_samples")
  # lambda_s[:,n] = tmp
  plotname = dirname2 * "prior_" * string(n) * ".pdf"
  scatter(lambda_s[1:n_samples_plot,n],alpha_s[1:n_samples_plot,n],legend=false,
            markershape=:circle,markersize=5.5,markeralpha=0.7,markercolor=:grey,markerstrokewidth=0.0,
            xlims=(0,35),xtickfont=font(20,"Times"),ylims=(0,1),ytickfont=font(20,"Times"));
  scatter!([λ],[α],legend=false,marker=(:star4,12.0,1.0,:black));
  savefig(plotname)
end
