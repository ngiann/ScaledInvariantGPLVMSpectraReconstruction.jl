##############################################################################################
function ppca(Y::Matrix{T}, σ::Matrix{T}; Q = Q, iterations = 1, seed = 1) where T<:Real
##############################################################################################

    rng = MersenneTwister(seed)

    D, N = size(Y)

    p₀ = randn(rng, numberofparameters_ppca(D, N, Q))

    ppca(Y, σ, p₀; Q = Q, iterations = iterations)

end


##############################################################################################
function ppca(Y::Matrix{T}, σ::Matrix{T}, p₀::Vector{T}; Q = 2, iterations = 1) where T<:Real
##############################################################################################

    #------------------------------------------------------------
    # Check dimensions and preliminaries
    #------------------------------------------------------------

    D, N = size(Y); @assert(size(Y) == size(σ))

    numparam = numberofparameters_ppca(D, N, Q)
    
    @printf("ppac: There are %d data items of dimension %d\n", N, D)
    @printf("Optimising %d number of free parameters\n", numparam)

    # Calculate observed diagonal covariance matrices

    S = σ.^2


    #------------------------------------------------------------
    function unpack_ppca(p)
    #------------------------------------------------------------

        local MARK = 0

        local W = reshape(p[MARK+1:MARK+D*Q], D, Q); MARK += D*Q

        local b = p[MARK+1:MARK+D]; MARK += D

        local c = softplus.(p[MARK+1:MARK+N]); MARK += N

        local β = softplus(p[end]); MARK += 1

        @assert(length(p) == MARK)

        return W, b, c, β

    end


    #------------------------------------------------------------
    function marginaloglikelihood(W, b, c, β)
    #------------------------------------------------------------

        local aux = zero(eltype(W))

        local WWᵀ = Symmetric(W*W')

        for n in 1:N

            aux += logpdf(MvNormal(c[n]*b, Diagonal(S[:,n]) + I/β + c[n]*c[n]*WWᵀ), Y[:,n])

        end

        return aux

    end


    #------------------------------------------------------------
    function posterior(yₙ, Sₙ, W, b, cₙ, β)
    #------------------------------------------------------------

        local L = inv(Sₙ + I/β)

        local Σpost⁻¹ = Symmetric(I + (cₙ^2)*W'*L*W) # see 2.117 in PRML

        local μpost = Σpost⁻¹\(cₙ*W'*L*(yₙ - cₙ*b))

        return μpost, Symmetric(inv(Σpost⁻¹))

    end


    #------------------------------------------------------------
    function reconstruct(B, ϕ, σ, W, b, β, Z; retries = retries)
    #------------------------------------------------------------

        # verify dimensions
        @assert(length(ϕ) == length(σ))
        @assert(size(B, 1) == length(ϕ))
        @assert(size(B, 2) == D)
    
        local S = Diagonal(σ.^2)

        function unpack(p)
            @assert(length(p) == Q+1)
            softplus(p[1]), p[2:end]
        end

        recobjective(c, z) = logpdf(MvNormal(c*B*(W*z + b), S + I/β), ϕ)
        
        local helper(p) = - recobjective(unpack(p)...)  

        local opt = Optim.Options(iterations=100_000)
   
        local res = [optimize(helper, [invsoftplus(1); Z[rand(1:N)]], NelderMead(), opt) for _ in 1:retries]

        local bestindex = argmin([r.minimum for r in res])

        local copt, zopt = unpack(res[bestindex].minimizer)
        
        # retunrs reconstructed object and latent coordinate
        return copt*(W*zopt + b), zopt

    end


    #------------------------------------------------------------
    # Setup and solve optimisation problem
    #------------------------------------------------------------

    helper(p) = -marginaloglikelihood(unpack_ppca(p)...)

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    res = optimize(helper, p₀, ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))


    #------------------------------------------------------------
    # Return coordinates X and optimised parameter vector
    #------------------------------------------------------------

    X, rec, reconstuct, c = let

        local W, b, c, β = unpack_ppca(res.minimizer)

        local Z = [posterior(Y[:,n], Diagonal(S[:,n]), W, b, c[n], β)[1] for n in 1:N]
        
        local rec = reduce(hcat, [c[n]*(W*Z[n] + b) for n in 1:N])

        reduce(hcat, Z), rec, (B, ϕ, σ; retries = 10) -> reconstruct(B, ϕ, σ, W, b, β, Z; retries = retries), c

    end
    
    X, res.minimizer, rec, reconstuct, res.minimum, c

end


#------------------------------------------------------------
numberofparameters_ppca(D, N, Q) = D*Q + D + N + 1
#------------------------------------------------------------