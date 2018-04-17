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
