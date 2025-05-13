#-----------------------------------------------------
@stable function lowerbound(Y, S, μ, λ, β, X, ν, τ, w, θ)::Float64
#-----------------------------------------------------
                              
    D, N = size(Y)

    # numobservations = sum(!isinf, Y)

    @assert(length(ν) == length(τ) == N); @assert(size(X, 2) ==  N)

    K = calculateK(X, θ)

    Λ = Diagonal(λ)

    Σ = Symmetric(inv(Λ)*((inv(Λ) + K)\K)) #calculateposteriorcov(K, λroot)

    aux = zero(eltype(μ)) # - 0.5*numobservations*log(2π)

    @inbounds for n in 1:N, d in 1:D

        if isinf(Y[d, n])
            continue
        end
                    
        ς² = S[d,n] + 1/β
 
        aux += logpdf(Normal(μ[d, n] * g(ν[n], τ[n]), sqrt(ς²)), Y[d, n]) - (0.5/ς²) * μ[d, n]^2 * v(ν[n], τ[n]) - (0.5/ς²) * Σ[n,n] * h(ν[n], τ[n])

        # aux += - 0.5*log(ς²)

        # aux += - 0.5*(1/ς²)*abs2(c[n]*m - Y[d, n])

        # aux += - 0.5*(1/ς²)*c[n]*c[n]*v

        

    end

    aux += expectation_sum_D_log_prior(μ = μ, Σ = Σ, K = K)
    
    aux += D*gaussianentropy(Σ) 

    # entropy of scaling posterior
    for n in 1:N

        aux += trunc_entropy(ν[n], τ[n])

    end

    for n in 1:N

        aux += trunc_expectation_logN(1, 0, 1000.0, ν[n], τ[n]) ####################

    end
    
    aux += quadratic_penalty(w, α = 1e-0)

    aux += quadratic_penalty(X, α = 1e-0)

    return aux 

end



#-----------------------------------------------------
function lowerbound_slow(Y, S, backend, μ, λ, β, X, c, w, θ) 
#-----------------------------------------------------

    D, N = size(μ)
        
    @assert(size(Y) == size(μ)); @assert(length(c) == N); @assert(size(X, 2) ==  N)


    K = calculateK(X, θ)

    C = Diagonal(c)

    Λ = Diagonal(λ)

    Σ = Symmetric(inv(Λ)*((inv(Λ) + K)\K)) #calculateposteriorcov(K, λroot)


    aux = zero(eltype(μ))

    
    for d in 1:D

        for n in 1:N
            
            σᵦ =  sqrt((S[d,n]) + I/β)

            m = M(backend, a=1, b=0, μ=μ[d,n], σ=sqrt(Σ[n,n]))

            v = V(backend, a=1, b=0, μ=μ[d,n], σ=sqrt(Σ[n,n]))
            
            aux += logpdf(Normal(C[n,n]*m, σᵦ), Y[d,n])
            
            aux += -0.5*(C[n,n] * C[n,n] * (1/σᵦ^2) * v)
            
        end

        aux += logpdf(MvNormal(zeros(N), K), μ[d,:])

        aux += -0.5*tr(K \ Σ)

        aux += entropy(MvNormal(zeros(N), Σ))

    end

    aux += quadratic_penalty(w, α = 1e-0) + quadratic_penalty(X, α = 1e-0)

    return aux 

end

