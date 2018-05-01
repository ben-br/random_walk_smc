# functions for Gibbs updates of non-SMC components


function calculatePredictiveParams(s_state::SamplerState,B_sum::Int64,K_sum::Int64,n_obs::Int64)::Tuple{Float64,Float64,Float64,Float64}

    ap_alpha = s_state.a_α[1] + B_sum::Float64 # a in BetaBern(a,b)
    bp_alpha = s_state.b_α[1] + n_obs - B_sum::Float64 # b in BetaBern(a,b)
    ap_lambda = s_state.a_λ[1] + K_sum::Float64 # r in NB(r,p)
    bp_lambda = 1/(s_state.b_λ[1] + n_obs + 1)::Float64 # p in NB(r,p)

    return ap_alpha,bp_alpha,ap_lambda,bp_lambda

end

function updateBandK!(L::Array{Float64,2},s_state::SamplerState,particle_container::Array{Array{ParticleState,1},1},particle_path::Array{Int64,1})

  T = s_state.ne_data[1]
  k_range = s_state.k_range
  b_range = s_state.b_range
  k_trunc = length(k_range)

  # sufficient statistics
  # assert( s_state.K[1] == s_state.B[1] == zero(Int64) )
  K_sum = sum(s_state.K)
  B_sum = sum(s_state.B)

  t_order = randperm(T)

  # pre-allocate some things
  edge = zeros(Int64,2)
  rwp = zeros(Float64,2)
  frwp = zeros(Float64,1)
  eig_pgf = zeros(Float64,s_state.nv_data)
  root_vtx = zeros(Int64,1)

  p_k = zeros(Float64,k_trunc)
  lp_k = zeros(Float64,k_trunc)
  p_b = zeros(Float64,2)
  lp_b = zeros(Float64,2)

  for t in t_order

    if t>1 # skip t=1, deterministically inserts an edge with a new vertex

      # update sufficient statistics, predictive parameters
      K_sum += -s_state.K[t]
      B_sum += -s_state.B[t]
      ap_alpha,bp_alpha,ap_lambda,bp_lambda = calculatePredictiveParams(s_state,B_sum,K_sum,T-2)

      particle_tm1 = particle_container[particle_path[t-1]][t-1]
      particle_t = particle_container[particle_path[t]][t]
      nv_tm1 = particle_tm1.n_vertices[1]

      if particle_t.new_vertex[1]

        # update B_t
        root_vtx[:] = find(particle_tm1.vertex_unmap[particle_t.edge_list[t,:]] .> 0)::Array{Int64,1}
        # assert(countnz(root_vtx)==1)
        if !particle_tm1.has_eigensystem[1]
          updateEigenSystem!(L,nv_tm1,particle_tm1)
          L[:] = zero(Float64)
        end
        nbPred!(eig_pgf,nv_tm1,ap_lambda,bp_lambda,particle_tm1.eig_vals)
        # get zero-one ball
        nbd = [root_vtx; particle_tm1.nbd_list[root_vtx,1:particle_tm1.degrees[root_vtx]]]::Array{Float64,1}
        # calculate fruitless random walk probs
        frwp[:] = randomWalkProbs(root_vtx,nbd,nv_tm1,particle_tm1.degrees,eig_pgf,particle_tm1.eig_vecs)::Array{Float64,1}

        lp_b[2] = log(ap_alpha) - log(ap_alpha + bp_alpha)
        lp_b[1] = log(bp_alpha) - log(ap_alpha + bp_alpha) + log(frwp[1])
        logSumExpWeightsNorm!(p_b,lp_b)
        s_state.B[t] = wsample[b_range,p_b]

        # update K_t
        for k in k_range
          for i = 1:nv_tm1
            k==1 ? eig_pgf[i] = particle_tm1.eig_vals[i] : eig_pgf[i] *= eig_pgf[i]
          end
          frwp[:] = randomWalkProbs(root_vtx,nbd,nv_tm1,particle_tm1.degrees,eig_pgf,particle_tm1.eig_vecs)
          lp_k[k] = lgamma(k + ap_lambda) - lgamma(k + 1) +
                      log( ap_alpha/(ap_alpha + bp_alpha) + (bp_alpha/(ap_alpha + bp_alpha))*frwp[1] )
        end
        logSumExpWeightsNorm!(p_k,lp_k)
        s_state.K[t] = wsample(k_range,p_k) - 1

        # update sufficient statistics
        B_sum += s_state.B[t]
        K_sum += s_state.K[t]

      else # new edge w/out vertex
        s_state.B[t] = zero(Int64)

        # update K_t
        if !particle_tm1.has_eigensystem[1]
          updateEigenSystem!(L,nv_tm1,particle_tm1)
          L[:] = zero(Float64)
        end

        for k in k_range
          for i = 1:nv_tm1
            k==1 ? eig_pgf[i] = particle_tm1.eig_vals[i] : eig_pgf[i] *= eig_pgf[i]
          end

          edge[:] = particle_t.vertex_unmap[particle_t.edge_list[t,:]]
          randomWalkProbs!(rwp,edge,nv_tm1,particle_tm1.degrees,eig_pgf,particle_tm1.eig_vecs)::Array{Float64,1}

          lp_k[k] = lgamma(k + ap_lambda) - lgamma(k + 1) +
                      log(bp_alpha) - log(ap_alpha + bp_alpha) +
                      log( particle_tm1.degrees[edge[1]]*rwp[1] + particle_tm1.degrees[edge[2]]*rwp[2] )
        end
        logSumExpWeightsNorm!(p_k,lp_k)
        s_state.K[t] = wsample(k_range,p_k) - 1

        # update sufficient statistics
        B_sum += s_state.B[t]
        K_sum += s_state.K[t]

      end

    end

  end

end

function updateAlphaAndLambda!(s_state::SamplerState)

  B_sum = sum(s_state.B)
  K_sum = sum(s_state.K)
  s_state.α[:] = rand(Beta(s_state.a_α + B_sum, s_state.b_α + s_state.ne_data - 1 - B_sum))
  s_state.λ[:] = rand(Gamma(s_state.a_λ + K_sum, 1/(s_state.b_λ + s_state.ne_data - 1)))

end


function saveSamples!()

end
