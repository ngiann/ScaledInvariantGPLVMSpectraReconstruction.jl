function gplvmpredictive(; net = net, res = res, Q = Q, N = N, D = D)

    λ, β, X, w, θ = unpack_gplvm(res.minimizer, net, Q, N)


    # calculate posterior mean and covariance

    K = calculateK(X, θ) # N × N

    Λ = Diagonal(λ) # N × N

    μ = net(w, X); # D × N

    K⁻¹μ = K\μ'


    #-------------------------------------------------------------------------

    function predictiveposterior(x::Vector)

        local Kx = calculateK(reshape(x, Q, 1), X, θ) # Nx × N

        local Kxx = calculateK(reshape(x, Q, 1), reshape(x, Q, 1), θ) # Nx × Nx

        local m = (Kx * K⁻¹μ)'

        local P = (Kxx - Kx*((K + inv(Λ))\Kx'))

        return vec(m), only(P)

    end


    function predmean(x)

        predictiveposterior(x)[1]

    end


    function getloglikel(y::Vector, σ::Vector)

        function f(x::Vector)

            local m, σ²pred = predictiveposterior(x)

            return logpdf(MvNormal(m, Diagonal(vec(σ.^2 .+ 1.0./β .+ σ²pred))), vec(y)) + quadratic_penalty(x; α = 1e-0)
           
        end

    end


    function getloglikel(B, y, σ)

        function f(x)

            local m, σ²pred = predictiveposterior(x)
            
            return logpdf(MvNormal(B*m, Diagonal(σ.^2) + (1.0./β + σ²pred)*(B*B')), vec(y)) + quadratic_penalty(x; α = 1e-0)

        end

    end


    function unpk(p)

        local MARK = 0

        local x = reshape(p[MARK+1:MARK+1*Q], Q); MARK += Q*1

        @assert(length(p) == MARK) # make sure we used up all elements in p

        return x

    end


    function infer(y::Vector{T}, σ::Vector{T}; repeat = 30, seed = 1) where T<:Real

        local rng = MersenneTwister(seed)

        local ℓ = getloglikel(y, σ)

        local opt = Optim.Options(iterations = 10000, show_trace = true, show_every = 1)

        local helper = x -> - ℓ(unpk(x))

        local minoptfunc() = Optim.optimize(helper, vec(X[:,rand(rng, 1:N)]), LBFGS(), opt, autodiff = AutoMooncake(config = nothing))

        return unpk(repeatoptimisation(minoptfunc, repeat))

    end


    function infer(B, y::Vector{T}, σ::Vector{T}; repeat = 30, seed = 1) where T<:Real

        local rng = MersenneTwister(seed)

        local ℓ = getloglikel(B, y, σ)

        local opt = Optim.Options(iterations = 100_000, show_trace = true, show_every = 1)
        
        local helper = x -> - ℓ(unpk(x))
     
        local minoptfunc() = Optim.optimize(helper, vec(X[:,rand(rng, 1:N)]), ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))
       
        return unpk(repeatoptimisation(minoptfunc, repeat))

    end


    function getvilogl(y::Vector{T}, σ::Vector{T}) where T<:Real

        @assert(size(y) == size(σ))

        local ℓ = getloglikel(y, σ)

        local freenp = Q

        return freenp, x -> ℓ(unpk(x))

    end


    function getvilogl(B, y::Vector{T}, σ::Vector{T}) where T<:Real

        @assert(size(y) == size(σ))

        local ℓ = getloglikel(B, y, σ)

        local freenp = Q

        return freenp, x -> ℓ(unpk(x))

    end

    #-------------------------------------------------------------------------


    return infer, getvilogl, predmean

end