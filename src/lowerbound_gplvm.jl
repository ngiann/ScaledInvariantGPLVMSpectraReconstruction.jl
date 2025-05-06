#-----------------------------------------------------
@stable function lowerbound_gplvm(Y, S, net, λ, β, X, w, θ)::Float64
#-----------------------------------------------------

    D, N = size(Y)

    numobservations = sum(!isinf, Y)

    @assert(size(X, 2) ==  N)

    K = calculateK(X, θ)

    μ = net(w, X); @assert(size(Y) == size(μ)); 

    Λ = Diagonal(λ)

    Σ = Symmetric(inv(Λ)*((inv(Λ) + K)\K)) #calculateposteriorcov(K, λroot)

    aux = - 0.5*numobservations*log(2π)

    @inbounds for n in 1:N, d in 1:D
                  
        if isinf(Y[d, n])
            continue
        end
        
        ς² = S[d,n] + 1/β

        aux += - 0.5*log(ς²)

        aux += - 0.5*(1/ς²)*abs2(μ[d, n] - Y[d, n])

        aux += - 0.5*(1/ς²)*Σ[n,n]

    end

    aux += expectation_sum_D_log_prior(μ = μ, Σ = Σ, K = K)
    
    aux += D*gaussianentropy(Σ) 
    
    aux += quadratic_penalty(w, α = 1e-0)

    aux += quadratic_penalty(X, α = 1e-0)

    return aux 

end



#-----------------------------------------------------
function lowerbound_gplvm_slow(Y, S, net, λ, β, X, w, θ) 
#-----------------------------------------------------

    D, N = size(Y)

    @assert(size(X, 2) ==  N)

    K = calculateK(X, θ)

    μ = net(w, X); @assert(size(Y) == size(μ)); 

    Λ = Diagonal(λ)

    Σ = Symmetric(inv(Λ)*((inv(Λ) + K)\K)) #calculateposteriorcov(K, λroot)


    aux = zero(eltype(μ))


    for d in 1:D

        Sᵦ =  Diagonal(S[d,:]) + I/β

        aux += logpdf(MvNormal(μ[d,:], Sᵦ), Y[d,:])

        aux += -0.5*tr(inv(Sᵦ) * Σ)

        aux += logpdf(MvNormal(zeros(N), K), μ[d,:])

        aux += -0.5*tr(K \ Σ)

        aux += entropy(MvNormal(zeros(N), Σ))

    end

    aux += quadratic_penalty(w, α = 1e-0) + quadratic_penalty(X, α = 1e-0)

    return aux 

end

