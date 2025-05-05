function gaussianentropy(Σ)

    N = size(Σ, 1)

    U = cholesky(Σ).L

    0.5*N*log(2*π*ℯ) + 0.5*(2*sum(log.(diag(U)))) # 0.5*logabsdet(Σ)[1]

end