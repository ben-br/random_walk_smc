# smc utilities

function uniqueParticles!(unq::Array{Int64,2},t::Int64,ed_idx::Array{Int64,1},ancestors::Array{Int64,2})

  nunq = zero(Int64)
  anc = ancestors[t,:]
  for n = 1:length(ed_idx)
    if t==1
      root_pos = findfirst( ed_idx .== ed_idx[n] )
    else
      root_pos = findfirst( ( ed_idx .== ed_idx[n] ).*( unq[t-1,anc] .== unq[t-1,anc[n]] ) )
    end

    if root_pos == n
      nunq += one(Int64)
      unq[t,n] = nunq
    else
      unq[t,n] = unq[t,root_pos]
    end
  end

end

function randomWalkProbs!(W::Array{Float64,2},nv::Int64,degrees::Array{Float64,1},eig_pgf::Array{Float64,1},esys_vec::Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}})
  s = zeros(Float64,1)
  for n = 1:nv # column index
    for m = 1:n # row index
      s[1] = zero(Float64)
      for k = 1:nv
        s[1] += esys_vec[k,m] * esys_vec[k,n] * eig_pgf[k]
      end
      W[m,n] = s[1]*sqrt(degrees[n]/degrees[m])
      n==m ? nothing : W[n,m] = s[1]*sqrt(degrees[m]/degrees[n])
    end

  end
  elMax!(W,zero(Float64)) # for numerical stability
  elMin!(W,one(Float64)) # for numerical stability
end

function randomWalkProbs!(w::Array{Float64,2},vtx_pairs::Array{Int64,2},nv::Int64,degrees::Array{Float64,1},
    eig_pgf::Array{Float64,1},esys_vec::Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}})
"""
  Calculate random walk probabilities for vertex pairs in `vtx_pairs`
"""

  # w = zeros(Float64,size(vtx_pairs,1),2)
  resetArray!(w)
  for i in 1:size(vtx_pairs,1)
    for k in 1:nv
      w[i,1] += (esys_vec[k,vtx_pairs[i,1]] * esys_vec[k,vtx_pairs[i,2]] * eig_pgf[k])::Float64
    end

    w[i,2] = w[i,1]*(degrees[vtx_pairs[i,1]]/degrees[vtx_pairs[i,2]])^(0.5)::Float64
    w[i,1] *= ((degrees[vtx_pairs[i,2]]/degrees[vtx_pairs[i,1]])^(0.5))::Float64

    elMax!(w,zero(Float64)) # for numerical stability
    elMin!(w,one(Float64)) # for numerical stability

  end

end

function randomWalkProbs(root_vtx::Int64,neighbors::Array{Int64,1},nv::Int64,degrees::Array{Float64,1},eig_pgf::Array{Float64,1},esys_vec::Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}})::Float64
"""
  Calculate random walk probabilities for vertex pairs in `vtx_pairs`
"""

  w = zero(Float64)
  z = zeros(Float64,1)
  for i in 1:convert(Int64,degrees[root_vtx])
    z[:] = zero(Float64) # reset z
    for k in 1:nv
      z[1] += (esys_vec[k,root_vtx] * esys_vec[k,neighbors[i]] * eig_pgf[k])::Float64
    end
    z[1] *= sqrt(degrees[neighbors[i]]/degrees[root_vtx])::Float64
    w += z[1]::Float64
  end
  w = max(w,zero(Float64)) # for numerical stability
  w = min(w,one(Float64)) # for numerical stability
  return w

end

function fruitlessRWProb(W::Array{Float64,2},nbd_list::Array{SparseVector{Int64,Int64},1},root_vtx::Int64)::Float64

  p = zero(Float64)
  p += W[root_vtx,root_vtx]::Float64

  rnz = nbd_list[root_vtx].nzval
  for i in rnz
    p += W[root_vtx,i]
  end
  return p
end

function fruitlessRWProb(W::Array{Float64,2},nbd_list::Array{Int64,2},root_vtx::Int64)::Float64

  p = zero(Float64)
  p += W[root_vtx,root_vtx]::Float64

  i = 1
  while nbd_list[root_vtx,i] > zero(Float64)
    p += W[root_vtx,nbd_list[root_vtx,i]]
    i += 1
  end

  return p
end

function fruitlessRWProb(W::Array{Float64,2},adj::Array{Int64,2},root_vtx::Int64,nv::Int64)::Float64

  # find neighborhood of root_vtx
  # nbd = find(adj[:,root_vtx])::Array{Float64,1}

  p = zero(Float64)
  p += W[root_vtx,root_vtx]::Float64

  for i = 1:nv
    adj[i,root_vtx]==one(Float64) ? p += W[root_vtx,i] : nothing
  end
  return p
end

function nbPred!(pgf::Array{Float64,1},nv::Int64,a_lambda::Float64,b_lambda::Float64,evals::Union{Array{Float64,1},SparseVector{Float64,Int64}})

    for i=1:nv
      pgf[i] = ((1.0 - b_lambda)^(a_lambda))*( (1.0 .- evals[i])*((1.0 - b_lambda*(1.0 - evals[i])).^(-a_lambda)) )
      isinf(pgf[i]) ? pgf[i] = zero(Float64) : nothing # Moore-Penrose pseudo-inverse
    end

end
