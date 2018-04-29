# functions for Gibbs updates of non-SMC components


function calculatePredictiveParams(s_state::SamplerState,B_sum::Int64,K_sum::Int64,n_obs::Int64)::Tuple{Float64,Float64,Float64,Float64}

    ap_alpha = s_state.a_α[1] + B_sum::Float64
    bp_alpha = s_state.b_α[1] + n_obs - B_sum::Float64
    ap_lambda = s_state.a_λ[1] + K_sum::Float64
    bp_lambda = 1/(s_state.b_λ[1] + n_obs + 1)::Float64

    return ap_alpha,bp_alpha,ap_lambda,bp_lambda

end

function updateBandK!(s_state::SamplerState,particle_container::Array{Array{ParticleState,1},1},particle_path::Array{Int64,1})

    T = length(particle_container[1])
    k_range = 0:s_state.k_trunc
    b_range = 0:1

    # sufficient statistics
    K_sum = sum(s_state.K)
    B_sum = sum(s_state.B)

    # parameters of posterior predictive distributions
    # ap_lambda = s_state.a_λ[1] + K_sum::Float64 # r in NB(r,p)
    # bp_lambda = 1/(s_state.b_λ[1] + T)::Float64 # p in NB(r,p)
    #
    # ap_alpha = s_state.a_α[1] + B_sum::Float64 # a in BetaBern(a,b)
    # bp_alpha = s_state.b_α[1] + T - B_sum::Float64 #b in BetaBern(a,b)

    t_order = randperm(T)
    # pre-allocate some things
    lGammaAll = zeros(Float64,s_state.k_trunc+1)
    lGammaK = zeros(Float64,s_state.k_trunc+1)
    lGammaAP = zeros(Float64,1)
    p_k = zeros(Float64,s_state.k_trunc+1)
    lp_k = zeros(Float64,s_state.k_trunc+1)

    p_b = zeros(Float64,2)
    lp_b = zeros(Float64,2)

    for t in t_order

        if t==1

            # update sufficient statistics, predictive parameters
            K_sum += -s_state.K[t]
            B_sum += -s_state.B[t]

            ap_alpha,bp_alpha,ap_lambda,bp_lambda = calculatePredictiveParams(s_state,B_sum,K_sum,T-1) # excludes current step

            # fruitless random walk probability
            frwp = one(Float64)

            # new vertex probabilty
            lp_b[2] = log(ap_alpha) - log(ap_alpha + bp_alpha)
            lp_b[1] = log(bp_alpha) - log(ap_alpha + bp_alpha) + log(frwp)

            # update B_t
            logSumExpWeightsNorm!(p_b,lp_b)
            s_state.B[t] = wsample(b_range,p_b)
            B_sum += s_state.B[t]

            # random walk probabilities
            # gamma functions
            lGammaAll[:] = lgamma.(ap_lambda .+ k_range)
            lGammaK[:] = lgamma.(k_range .+ 1)
            lGammaAP[:] = lgamma(ap_lambda)

            lp_k[:] = lGammaAll .- lGammaK .- lGammaAP .+ ap_lambda.*log.(1 - bp_lambda) .+ k_range.*log(bp_lambda) .+
                    log.(ap_alpha .+ bp_alpha.*frwp) .- log(ap_alpha + bp_alpha)

            # update K_t
            logSumExpWeightsNorm!(p_k,lp_k)
            s_state.K[t] = wsample(1:(s_state.k_trunc+1),p_k)
            K_sum += s_state.K[t]

        else

            

        end



    end


end

function updateAlpha!()

end

function updateLambda!()

end

function saveSamples!()

end
