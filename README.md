# LoL-team-embedding

[League of Leagends](https://euw.leagueoflegends.com/en-gb/) is a 5v5 team game where players can choose to play as one of 149 champions. Since each player tends to perform one of 5 roles there are 6.862560348𝐸+10 possible team compositions. 

Can these team compositions be better understood though an embedding into a lower dimensional space? And can this new axis be used complete winning team compositions or suggest couter compositions? (Particularly as a pre-trained initial layer in a deeped network)

I initially obtained data from the riot API with [Cassiopeia](https://cassiopeia.readthedocs.io/en/latest/), starting with games by challenger (highest tier) players, followed by master tier players. Despite having a substantial amount of training data (~50K teams), both the autoencoder and CBoW models were over-generalising - they were predicting a single champion in all contexts.

Examining the distribution of champions in each role shows that some are picked more often than others (excpected) however, the top champions were being chosen ~1.5-2x as often as the next most popular. I downsampled to achive a more even distribution at the top end while leaving the overall shape. Unfortunately this did not solve the problem, neither did small batches or even more epochs.

Fortunately focal loss porvides the weighting the loss needs to reach deeper minima and start to predict in a context dependant way. 
