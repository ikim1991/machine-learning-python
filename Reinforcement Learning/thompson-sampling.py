# Thompson Sampling

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load Dataset
dataset = pd.read_csv('./dataset/Ads_CTR_Optimisation.csv')

# Implenting Thompson Sampling
N = 10000
d = 10

# Initializing List of Ads Selected in each round
ads_selected = []

# Initializing rewards of algorithm
numbers_of_rewards_0 = [0] * d
numbers_of_rewards_1 = [0] * d
total_reward = 0

# Simulating the 10,000 Rounds as seen in the dataset
for n in range(0,N):
  # Initialize ad with highest probability of success
  ad = 0
  max_random = 0
  # Loop through each ad and select the ad with the highest probability of success
  for i in range(0,d):
    random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
    # Selects the ad with the highest upper confidence bound
    if random_beta > max_random:
      max_random = random_beta
      ad = i
  # List of the ad selected at each round
  ads_selected.append(ad)
  # Reward based on the simulated dataset, with a reward being awarded when the user clicked on the ad
  reward = dataset.values[n, ad]
  # Number of times each ad was selected
  if reward == 1:
      numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
  else:
      numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1

  # Total reward of the algorithm
  total_reward = total_reward + reward

# Visualizing the results
plt.hist(ads_selected)
plt.title('Thompson Sampling Ad Selection Distribution')
plt.xlabel('Ad Selected')
plt.ylabel('Number of Times Selected')
