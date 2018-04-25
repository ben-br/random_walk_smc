# utilities

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

  shifted_weights = exp.(log_w .- max_entry)

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

function adj2edgelist(A::T where T<:Union{Array{Int64,2},SparseMatrixCSC{Int64,Int64}})
  adj2edgelist(A,false)
end

function adj2edgelist(A::T where T<:Union{Array{Int64,2},SparseMatrixCSC{Int64,Int64}},multigraph::Bool)
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

function edgelist2adj(edgelist::Array{Int64,2})
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

function normalizedLaplacian(edgelist::Array{Int64,2})
"""
  Construct (sparse) normalized Laplacian matrix
"""
  degrees = getDegrees(edgelist)
  nv = maximum(edgelist)
  L = sparse(1*I,nv,nv) - sparse(Diagonal( degrees.^(-1/2) )) * edgelist2adj(edgelist,Float64) * sparse(Diagonal( degrees.^(-1/2) ))
end

function getDegrees(edgelist::Array{Int64,2})
"""
  Compute degrees of edge list
"""
  return tally_ints(edgelist[:],maximum(edgelist))
end
