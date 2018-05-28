ENV["JULIA_PKGDIR"] = "/data/ziz/bloemred/.julia"  #set package directory
using JLD    #run Pkg.add("JLD") if you do not have relevant packages.
using Iterators # package that can be used to generates hyperparameter grid

# hyperparameter values in sweep
a, b, c = linspace(0.001,0.1,5),linspace(0.001,0.1,5),linspace(0.01,0.5,5)
hyperparams = Iterators.product(a,b,c)
numhyperparams = length(hyperparams)

job_id = parse(Int32,ENV["SLURM_ARRAY_JOB_ID"])    #parse environment variables in Julia
task_id = parse(Int32,ENV["SLURM_ARRAY_TASK_ID"])

if task_id > numhyperparams
  error("Number of jobs in array is more than number of hyperparameters")
end

hyperparam = Iterators.nth(hyperparams,task_id)
@show job_id
@show task_id
@show hyperparam

#Now we can run our function with the hyperparameter, for example
function myfunc(hyp)
        a,b,c=hyp
        sleep(10)   # sleep for 10 seconds
        return(a+b+c)
end

out=myfunc(hyperparam)
dict=Dict("out"=>out,"hyperparameter"=> hyperparam)
save("/data/localhost/not-backed-up/bloemred/jobname_$(job_id)_$(task_id).jld","dict",dict)   #save to localhost(ziz01-ziz08)
