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

        local Σpost⁻¹ = Symmetric(I + cₙ^2*W'*L*W) # see 2.117 in PRML

        local μpost = Σpost⁻¹\(cₙ*W'*L*(yₙ - cₙ*b))

        return μpost, Symmetric(inv(Σpost⁻¹))

    end


    #------------------------------------------------------------
    # Setup and solve optimisation problem
    #------------------------------------------------------------

    helper(p) = -marginaloglikelihood(unpack_ppca(p)...)

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    res = optimize(helper, p₀, ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))


    X = let

        local W, b, c, β = unpack_ppca(res.minimizer)

        local X = [posterior(Y[:,n], Diagonal(S[:,n]), W, b, c[n], β)[1] for n in 1:N]
        
        reduce(hcat, X)

    end

    X, res.minimizer
    
end


#------------------------------------------------------------
numberofparameters_ppca(D, N, Q) = D*Q + D + N + 1
#------------------------------------------------------------