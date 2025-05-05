@stable function quadratic_penalty(x; α = α)

    -0.5*α*mapreduce(abs2, +, x)

end