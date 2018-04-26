# utilities

function elMax!{S<:Real,T<:Real}(x::Array{S},y::T)
  x[:] = max.(x,y)
end


function elMin!{S<:Real,T<:Real}(x::Array{S},y::T)
  x[:] = min.(x,y)
end


function logSumExpWeightsNorm(log_w::Vector{T} where T <: AbstractFloat)
  # function uses log-sum-exp trick to compute normalized weights of a
  # vector of numbers stored as logs, avoiding underflow
  max_entry = maximum(log_w)

  shifted_weights = log_w - max_entry
  normalized_weights = exp.(shifted_weights)./sum(exp.(shifted_weights))

end

function logSumExpWeights(log_w::Vector{T} where T <: AbstractFloat)
  # function uses log-sum-exp trick to compute UNnormalized weights of a
  # vector of numbers stored as logs, avoiding underflow
  max_entry = maximum(log_w)

  shifted_weights = exp.(log_w .- max_entry)

end

function logSumExpWeights!(lse_w::Vector{T},log_w::Vector{T}) where T <: AbstractFloat
  # function uses log-sum-exp trick to compute UNnormalized weights of a
  # vector of numbers stored as logs, avoiding underflow
  max_entry = maximum(log_w)

  lse_w[:] = exp.(log_w .- max_entry)

end

function OneHot(n::Int64,k::Int64)

  onehotvector = spzeros(n)
  onehotvector[k] = 1
  return onehotvector

end


function stratifiedResample(weights::Vector{Float64},n_resample::Int64)::Vector{Int64}
  sample_idx = zeros(Int64,n_resample)
  stratifiedResample!(sample_idx,weights,n_resample)
  return sample_idx
end

function stratifiedResample!(sample_idx::Vector{Int64},weights::Vector{Float64},n_resample::Int64)
"""
 Implements stratified resampling as described in Appendix B of
 Fearnhead and Clifford (2003), JRSSB 65(4) 887--899

 weights is a vector of weights; n_resample is the number of resampled
 indices to return
"""
  norm_weights = cumsum(weights./sum(weights)) # normalize
  buckets = collect(0.0:(1.0/n_resample):1.0)[1:n_resample]
  locs = rand(n_resample)./n_resample .+ buckets

  sample_idx[:] = [ findfirst( locs[i] .<= norm_weights ) for i in 1:n_resample ]
  # return sample_idx
end

function systematicResample(weights::Vector{Float64},n_resample::Int64)::Vector{Int64}
  sample_idx = zeros(Int64,n_resample)
  systematicResample!(sample_idx,weights,n_resample)
  return sample_idx
end

function systematicResample!(sample_idx::Vector{Int64},weights::Vector{Float64},n_resample::Int64)

  norm_weights = cumsum(weights./sum(weights))
  buckets = collect(0.0:(1.0/n_resample):1.0)[1:n_resample]
  U = rand()*1.0/n_resample
  locs = (buckets .+ U)

  sample_idx[:] = [ findfirst( locs[i] .<= norm_weights ) for i in 1:n_resample ]
  # return sample_idx
end


function tally_ints(Z::Vector{Int},K::Int)
    """
    counts occurrences in `Z` of integers 1 to `K`
    - `Z`: vector of integers
    - `K`: maximum value to count occurences in `Z`
    """
    ret = zeros(Int,K)
    n = size(Z,1)
    idx_all = 1:n
    idx_j = trues(n)
    for j in 1:K
      for i in idx_all[idx_j]
        if Z[i]==j
          ret[j] += 1
          idx_j[i] = false
        end
      end
    end
    return ret
end

# function adj2edgelist(A::Array{Int64,2},multigraph::Bool)
# """
#   Convert binary adjacency matrix A to edge list
# """
#
#   M,N,V = findnz(triu(A))
#   if multigraph
#     return hcat(M,N),V
#   else
#     return hcat(M,N)
#   end
# end

function adj2edgelist(A::T where T<:Union{Array{Int64,2},SparseMatrixCSC{Int64,Int64}})::Array{Int64,2}
  adj2edgelist(A,false)
end

function adj2edgelist(A::T where T<:Union{Array{Int64,2},SparseMatrixCSC{Int64,Int64}},multigraph::Bool)::Array{Int64,2}
"""
  Convert binary adjacency matrix A to edge list
"""

  M,N,V = findnz(tril(A))
  if multigraph
    return hcat(M,N),V
  else
    return hcat(M,N)
  end
end

function edgelist2adj(edgelist::Array{Int64,2})::SparseMatrixCSC{Int64,Int64}
"""
  Convert edge list to binary adjacency matrix
"""

  edgelist2adj(edgelist,Int64)
end

function edgelist2adj(edgelist::Array{Int64,2},value_type::DataType)
"""
  Convert edge list to binary adjacency matrix, with non-zero values of type value_type
"""
  nv = maximum(edgelist)
  A = sparse([edgelist[:,1];edgelist[:,2]],[edgelist[:,2];edgelist[:,1]],ones(value_type,2*size(edgelist,1)),nv,nv)
end

function normalizedLaplacian(edgelist::Array{Int64,2})::SparseMatrixCSC{Int64,Int64}
"""
  Construct (sparse) normalized Laplacian matrix
"""
  degrees = getDegrees(edgelist)
  nv = maximum(edgelist)
  L = sparse(1*I,nv,nv) - sparse(Diagonal( degrees.^(-1/2) )) * edgelist2adj(edgelist,Float64) * sparse(Diagonal( degrees.^(-1/2) ))
end

function denseNormalizedLaplacian!(L::Array{Float64,2},adj::SparseMatrixCSC{Float64,Int64},degrees::Array{Float64,1},nv::Int64)
"""
  Construct a symmetric (dense) normalized Laplacian matrix
"""
  # degrees = getDegrees(edgelist)
  # nv = maximum(edgelist)
  L[1:nv,1:nv] = Array(sparse(1*I,nv,nv) - sparse(Diagonal( degrees.^(-1/2) )) * adj * sparse(Diagonal( degrees.^(-1/2) )))
end

function getDegrees(edgelist::Array{Int64,2})::Array{Float64,1}
"""
  Compute degrees of edge list
"""
  return convert(Array{Float64,1},tally_ints(edgelist[:],maximum(edgelist)))
end

function nbPred!(pgf::Array{Float64,1},a_lambda::Float64,b_lambda::Float64,evals::Array{Float64,1})

    pgf .= ((1.0 - b_lambda)^(a_lambda)).*( (1.0 .- evals).*((1.0 .- b_lambda.*(1.0 .- evals)).^(-a_lambda)) )

end
