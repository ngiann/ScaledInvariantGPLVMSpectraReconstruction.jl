function gplvmpredictive(; net = net, res = res, Q = Q, N = N, D = D)

    λ, β, X, w, θ = unpack_gplvm(res.minimizer, net, Q, N)


    # calculate posterior mean and covariance

    K = calculateK(X, θ) # N × N

    Λ = Diagonal(λ) # N × N

    μ = net(w, X); # D × N

    K⁻¹μ = K\μ'


    #-------------------------------------------------------------------------

    function predmean(x)

        local Kx = calculateK(x, X, θ) # Nx × N

        local Kxx = calculateK(x, x, θ) # Nx × Nx

        # calculate predictive based on posterior
        local m = (Kx * K⁻¹μ)'

        return m

    end

    
    function getloglikel(y, σ)

        function f(x)

            local Kx = calculateK(x, X, θ) # Nx × N

            local Kxx = calculateK(x, x, θ) # Nx × Nx

            # calculate predictive based on posterior
            local m = (Kx * K⁻¹μ)'

            local P = Symmetric(Kxx - Kx*((K + inv(Λ))\Kx'))
   
            local aux = quadratic_penalty(x; α = 1e-0)
            
            for d in 1:D
           
                aux += logpdf(MvNormal(m[d,:],  Diagonal(σ[d,:].^2 .+ 1.0./β) + P), y[d,:])
           
            end

            return aux
        end

    end


    function getloglikel(B, y, σ)

        function f(x)

            local Kx = calculateK(x, X, θ) # Nx × N

            local Kxx = calculateK(x, x, θ) # Nx × N

            # calculate predictive based on posterior
            local m = (Kx * K⁻¹μ)'

            local P = Symmetric(Kxx - Kx*((K + inv(Λ))\Kx'))
            
            local Bm = B*m
   
            local aux = quadratic_penalty(x; α = 1e-0)
            
            aux += logpdf(MvNormal(vec(Bm), Diagonal(vec(σ).^2) + Symmetric(kron((P + (I/β)), B*B'))), vec(y))

            return aux

        end

    end


    function unpk(p, Nx)

        local MARK = 0

        local x = reshape(p[MARK+1:MARK+Nx*Q], Q, Nx); MARK += Q*Nx

        @assert(length(p) == MARK) # make sure we used up all elements in p

        return x

    end


    function infer(y, σ; repeat = 30, seed = 1)

        local rng = MersenneTwister(seed)

        local Nx = size(y, 2)

        local ℓ = getloglikel(y, σ)

        local opt = Optim.Options(iterations = 10000, show_trace = true, show_every = 1)

        local helper = x -> - ℓ(unpk(x,Nx))

        local minoptfunc() = Optim.optimize(helper, vec(X[:,rand(rng, 1:N, Nx)]), LBFGS(), opt, autodiff = AutoMooncake(config = nothing))

        return unpk(repeatoptimisation(minoptfunc, repeat), Nx)

    end


    function infer(B, y, σ; repeat = 30, seed = 1)

        local rng = MersenneTwister(seed)

        local Nx = size(y, 2)

        local ℓ = getloglikel(B, y, σ)

        local opt = Optim.Options(iterations = 100_000, show_trace = true, show_every = 1)
        
        local helper = x -> - ℓ(unpk(x,Nx))
     
        local minoptfunc() = Optim.optimize(helper, vec(X[:,rand(rng, 1:N, Nx)]), ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))
       
        return unpk(repeatoptimisation(minoptfunc, repeat), Nx)

    end


    function getvilogl(y, σ)

        local Nx = size(y, 2); @assert(size(y) == size(σ))

        local ℓ = getloglikel(y, σ)

        local freenp = Q*Nx

        return freenp, x -> ℓ(unpk(x, Nx))

    end


    function getvilogl(B, y, σ)

        local Nx = size(y, 2); @assert(size(y) == size(σ))

        local ℓ = getloglikel(B, y, σ)

        local freenp = Q*Nx

        return freenp, x -> ℓ(unpk(x, Nx))

    end

    #-------------------------------------------------------------------------


    return infer, getvilogl, predmean

end