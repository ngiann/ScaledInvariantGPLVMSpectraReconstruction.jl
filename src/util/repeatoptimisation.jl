function repeatoptimisation(minoptfunction, times)
    
    solutions = [minoptfunction() for _ in 1:times]

    bestindex = argmin([s.minimum for s in solutions])

    solutions[bestindex].minimizer

end