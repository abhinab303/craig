import numpy as np
import pandas as pd

max_epoch = 20
tau = 0.5
scores = np.ones(max_epoch)*100
scores = []

for i in range(max_epoch):
    scores.append(2**i)

scores = np.array(scores)

last_score = 0
# for epoch, score in enumerate(scores):
#     epoch = epoch + 1
#     if tau < 0.001:
#         weight_i = 0
#     else:
#         weight_i = np.exp(-epoch/tau)
#     weighted_score = (1 - weight_i) * last_score + weight_i * score
#     last_score = weighted_score
#     print(f"Epoch {epoch}, Score: {score}, Scaled Score: {weighted_score}")


for epoch, score in enumerate(scores):
    epoch = epoch + 1

    if epoch == 1:
        weighted_score = score
    else:
        weighted_score = tau*score + (1-tau)*last_score
    last_score = weighted_score
    print(f"Epoch {epoch}, Score: {score}, Scaled Score: {weighted_score}")
