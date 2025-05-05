#-----------------------------------------------------
function marginalloglikelihood(Y, S, β, c, X, _μ, w, θ)
#-----------------------------------------------------
    
    D, N = size(Y)
        
    @assert(length(c) == N); @assert(size(X, 2) ==  N)


    C = Diagonal(c)

    CKC = C * calculateK(X, θ) * C

    aux = zero(eltype(θ))

    for d in 1:D

        aux += logpdf(MvNormal(zeros(N), Symmetric(CKC + Diagonal(S[d,:]) + I/β)), Y[d,:])

    end

    aux += quadratic_penalty(w, α = 1e-0) + quadratic_penalty(X, α = 1e-0)

    return aux 

end