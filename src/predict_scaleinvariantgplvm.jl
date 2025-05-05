function scaleinvariantgplvmpredictive(; net = net, res = res, Q = Q, N = N, D = D)

    μ, λ, β, c, X, w, θ = unpack_scaleinvariantgplvm(res.minimizer, net, Q, N)


    # calculate posterior mean and covariance

    K = calculateK(X, θ) # N × N

    Λ = Diagonal(λ) # N × N

    # μ = net(w, X); # D × N

    K⁻¹μ = K\μ'



    local priorc = Uniform(1e-3, 100.0)

    #-------------------------------------------------------------------------

    function predmean(x)

        local Kx = calculateK(x, X, θ) # Nx × N

        local Kxx = calculateK(x, x, θ) # Nx × Nx

        # calculate predictive based on posterior
        local m = (Kx * K⁻¹μ)'

        return m

    end

    
    function getloglikel(y, σ)

        function f(x, c)

            local C = Diagonal(c)

            local Kx = calculateK(x, X, θ) # Nx × N

            local Kxx = calculateK(x, x, θ) # Nx × N

            # calculate predictive based on posterior
            local m = (Kx * K⁻¹μ)'

            local P = Symmetric(Kxx - Kx*((K + inv(Λ))\Kx'))
   
            local aux = quadratic_penalty(x; α = 1e-0)
            
            for d in 1:D
           
                aux += logpdf(MvNormal(C*m[d,:],  Diagonal(σ[d,:].^2 .+ 1.0./β) + Symmetric(C*P*C)), y[d,:])
           
            end

            aux += loglikelihood(priorc, c)

            return aux
        end

    end


    function getloglikel(B, y, σ)

        function f(x, c)

            local C = Diagonal(c)

            local Kx = calculateK(x, X, θ) # Nx × N

            local Kxx = calculateK(x, x, θ) # Nx × N

            # calculate predictive based on posterior
            local m = (Kx * K⁻¹μ)'

            local BmC = B*(m*C)

            local CPC = C*Symmetric(Kxx - Kx*((K + inv(Λ))\Kx'))*C
   
            local aux = quadratic_penalty(x; α = 1e-0)
            
            aux += logpdf(MvNormal(vec(BmC), Diagonal(vec(σ).^2) + Symmetric(kron((CPC+(I/β)), B*B'))), vec(y))

            aux += loglikelihood(priorc, c)

            return aux
        end

    end


    function unpk(p, Nx)

        local MARK = 0

        local x = reshape(p[MARK+1:MARK+Nx*Q], Q, Nx); MARK += Q*Nx

        local c = softplus.(p[MARK+1:MARK+Nx]); MARK += Nx

        @assert(length(p) == MARK) # make sure we used up all elements in p

        return x, c

    end


    function infer(y, σ; repeat = 30, seed = 1)

        local rng = MersenneTwister(seed)

        local Nx = size(y, 2)

        local ℓ = getloglikel(y, σ)

        local opt = Optim.Options(iterations = 10000, show_trace = true, show_every = 1)

        local helper = x -> - ℓ(unpk(x,Nx)...)

        local minoptfunc() = Optim.optimize(helper, [vec(X[:,rand(rng, 1:N, Nx)]);invsoftplus.(ones(Nx))], LBFGS(), opt, autodiff = AutoMooncake(config = nothing))

        return unpk(repeatoptimisation(minoptfunc, repeat), Nx)

    end


    function infer(B, y, σ; repeat = 30, seed = 1)

        local rng = MersenneTwister(seed)

        local Nx = size(y, 2)

        local ℓ = getloglikel(B, y, σ)

        local opt = Optim.Options(iterations = 100_000, show_trace = true, show_every = 1)
        
        local helper = x -> - ℓ(unpk(x,Nx)...)

        local minoptfunc() = Optim.optimize(helper, [vec(X[:,rand(rng, 1:N, Nx)]);invsoftplus.(ones(Nx))], ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))
        # local minoptfunc() = Optim.optimi ze(helper, [vec(randn(Q,Nx));invsoftplus.(ones(Nx))], ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))

        return unpk(repeatoptimisation(minoptfunc, repeat), Nx)

    end


    # function inferseries(B, t, y, σ; repeat = 1, seed = 1, iterations = 1)

    #     local Nx = size(y, 2)

    #     # local res = [infer(B, y[:,[i]], σ[:,[i]]; repeat = repeat, seed = seed) for i in 1:Nx]

    #     # local X0 = reduce(hcat, [r[1] for r in res])
       
    #     # local c  = reduce(vcat, [r[2] for r in res])

    #     local net = create_rbfnet(M = 25, r = 0.1, Q = Q, tobs = t)

    #     # local W = X0 * net.Φ' / (net.Φ * net.Φ' + 1e-6*I)

    #     local ℓ = getloglikel(B, y, σ)

    #     local opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 5)
        
    #     local aux(p) = net(p[1:numparam(net)]), softplus.(p[numparam(net)+1:end])

    #     # return  ((Q * net.M) + Nx), x -> ℓ(aux(x)...)
    #     local helper = x -> - ℓ(aux(x)...)

    #     # local minoptfunc() = Optim.optimize(helper, [vec(W); invsoftplus.(c)], ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))


    #     local minoptfunc() = Optim.optimize(helper, [vec(randn(Q,net.M)); invsoftplus.(ones(Nx))], ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))

    #     local uopt = repeatoptimisation(minoptfunc, repeat)
        
    #     local Wopt = reshape(uopt[1:numparam(net)], Q, net.M)

    #     return net(Wopt), get_t(net, Wopt, 0.1)

    # end



    function getvilogl(y, σ)

        local Nx = size(y, 2); @assert(size(y) == size(σ))

        local ℓ = getloglikel(y, σ)

        local freenp = Q*Nx + Nx

        return freenp, x -> ℓ(unpk(x, Nx)...)

    end


    function getvilogl(B, y, σ)

        local Nx = size(y, 2); @assert(size(y) == size(σ))

        local ℓ = getloglikel(B, y, σ)

        local freenp = Q*Nx + Nx

        return freenp, x -> ℓ(unpk(x, Nx)...)

    end


    #-------------------------------------------------------------------------


    return infer, getvilogl, predmean

end