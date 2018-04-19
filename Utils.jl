module Utils

function elMax!{S<:Real,T<:Real}(x::Array{S},y::T)
  x[:] = max.(x,y)
end


function elMin!{S<:Real,T<:Real}(x::Array{S},y::T)
  x[:] = min.(x,y)
end


function logSumExpWeightsNorm(log_w::Vector{T}) where T <: AbstractFloat
  # function uses log-sum-exp trick to compute normalized weights of a
  # vector of numbers stored as logs, avoiding underflow
  max_entry = maximum(log_w)

  shifted_weights = log_w - max_entry
  normalized_weights = exp.(shifted_weights)./sum(exp.(shifted_weights))

end

function logSumExpWeights(log_w::Vector{T}) where T <: AbstractFloat
  # function uses log-sum-exp trick to compute UNnormalized weights of a
  # vector of numbers stored as logs, avoiding underflow
  max_entry = maximum(log_w)

  shifted_weights = log_w - max_entry

end

function OneHot(n::Int64,k::Int64)

  onehotvector = spzeros(n)
  onehotvector[k] = 1
  return onehotvector

end


function stratifiedResample(weights::Vector{Float64},n_resample::Int64)
"""
 Implements stratified resampling as described in Appendix B of
 Fearnhead and Clifford (2003), JRSSB 65(4) 887--899

 weights is a vector of weights; n_resample is the number of resampled
 indices to return
"""
  norm_weights = weights./sum(weights) # normalize
  K = sum(norm_weights)/n_resample
  U = rand()*K

  sample_idx = zeros(Int64,n_resample)
  r = 0
  for n in 1:length(weights)
    U += -norm_weights[n]
    if U < 0.0
      r = r+1
      sample_idx[r] = n
      U += K
    end
  end
  return sample_idx
end
