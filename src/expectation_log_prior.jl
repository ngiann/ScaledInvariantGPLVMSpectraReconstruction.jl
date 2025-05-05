function expectation_log_prior(;μ = μ, K = K, Σ = Σ, b = 0.0)

    N = size(Σ, 1); @assert(size(Σ, 2) == N) # make sure it is square
                    @assert(size(Σ) == size(K))
                    @assert(length(μ) == N)

    U = cholesky(K).L

    # tr(U'\(U\Σ)) is equivalent to tr(K\Σ)
    # - sum(log.(diag(U))) is equivalent to -0.5*logdet(K)
    # U\μ' is equivalent to ...

    - 0.5*sum(abs2.(U\(μ.-b))) - 0.5*N*log(2π) - sum(log.(diag(U))) - 0.5*tr(U'\(U\Σ))

end

function expectation_sum_D_log_prior(;μ = μ, K = K, Σ = Σ)

    N = size(Σ, 1); @assert(size(Σ, 2) == N) # make sure it is square
    
    @assert(size(Σ) == size(K))

    D = size(μ, 1); @assert(size(μ, 2) == N)

    U = cholesky(K).L

    # tr(U'\(U\Σ)) is equivalent to tr(K\Σ)
    # - sum(log.(diag(U))) is equivalent to -0.5*logdet(K)
    # U\μ' is equivalent to ...

    - 0.5*sum(abs2.(U\μ')) - 0.5*D*N*log(2π) - D*sum(log.(diag(U))) - 0.5*D*tr(U'\(U\Σ))

end


# verification of method above
function TEST_expectation_sum_D_log_prior(;μ = μ, K = K, Σ = Σ, b = 0.0)

    N = size(Σ, 1); @assert(size(Σ, 2) == N) # make sure it is square
    @assert(size(Σ) == size(K))
    D = size(μ, 1); @assert(size(μ, 2) == N)

    aux = -0.5*D*tr((K\Σ))
    
    for d in 1:D
        
        aux += logpdf(MvNormal(zeros(N) .+ b, K), μ[d, :]) 

    end
    
    return aux

end