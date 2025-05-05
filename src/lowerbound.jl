#-----------------------------------------------------
@stable function lowerbound(Y, S, backend, μ, λ, β, c, X, w, θ)::Float64
#-----------------------------------------------------

    D, N = size(Y)

    numobservations = sum(!isinf, Y)

    @assert(length(c) == N); @assert(size(X, 2) ==  N)

    K = calculateK(X, θ)

    Λ = Diagonal(λ)

    Σ = Symmetric(inv(Λ)*((inv(Λ) + K)\K)) #calculateposteriorcov(K, λroot)

    aux = - 0.5*numobservations*log(2π)

    @inbounds for n in 1:N, d in 1:D

        if isinf(Y[d, n])
            continue
        end
                    
        ς² = S[d,n] + 1/β
 
        m = M(backend, a=1, b=0, μ=μ[d,n], σ=sqrt(Σ[n,n]))

        v = V(backend, a=1, b=0, μ=μ[d,n], σ=sqrt(Σ[n,n]))

        aux += - 0.5*log(ς²)

        aux += - 0.5*(1/ς²)*abs2(c[n]*m - Y[d, n])

        aux += - 0.5*(1/ς²)*c[n]*c[n]*v

    end

    aux += expectation_sum_D_log_prior(μ = μ, Σ = Σ, K = K)
    
    aux += D*gaussianentropy(Σ) 
    
    aux += quadratic_penalty(w, α = 1e-0)

    aux += quadratic_penalty(X, α = 1e-0)

    return aux 

end



#-----------------------------------------------------
function lowerbound_slow(Y, S, backend, μ, λ, β, c, X, w, θ) 
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

