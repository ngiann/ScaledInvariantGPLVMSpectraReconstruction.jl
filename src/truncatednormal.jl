ϕ(x) = normpdf(x)
Φ(x) = normcdf(x)

trunc_Z(μ, σ) = normccdf(-μ/σ) # ! note this is the complementary cumulative, i.e. 1 - Φ

log_trunc_Z(μ, σ) = normlogccdf(-μ/σ) # ! note this is the  complementary cumulative, i.e. log(1 - Φ)


trunc_expectation(μ, σ) = μ + ϕ(-μ/σ) * σ / trunc_Z(μ, σ)

g(μ, σ) = trunc_expectation(μ, σ)


trunc_variance(μ, σ) = σ^2 * (1 - μ/σ * ϕ(-μ/σ) / trunc_Z(μ, σ) - (ϕ(-μ/σ) / trunc_Z(μ, σ))^2)

v(μ, σ) = trunc_variance(μ, σ)

h(μ, σ) = trunc_variance(μ, σ) + g(μ, σ)^2


trunc_entropy(μ, σ) = log(sqrt(2*π*ℯ)*σ) + log_trunc_Z(μ, σ) + (-μ/σ)*ϕ(-μ/σ)/(2*trunc_Z(μ, σ))



trunc_logpdf(x, μ, σ) = -log(σ) + normlogpdf((x-μ)/σ) - log_trunc_Z(μ, σ)

trunc_expectation_logN(μ, σ, u, τ) = trunc_expectation_logN(1, μ, σ, u, τ)

trunc_expectation_logN(a, μ, σ, u, τ) = trunc_logpdf(a*trunc_expectation(u, τ), μ, σ)        - a^2 * (0.5/σ^2) * trunc_variance(u, τ)

expectation_logN(a, μ, σ, u, τ) = -log(σ) + normlogpdf((a*trunc_expectation(u, τ) - μ) / σ)  - a^2 * (0.5/σ^2) * trunc_variance(u, τ)

