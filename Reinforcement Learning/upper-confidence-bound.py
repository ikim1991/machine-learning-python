# Upper Confidence Bound

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load Dataset
dataset = pd.read_csv('./dataset/Ads_CTR_Optimisation.csv')

# Initializing number of rounds and ads
N = 10000
d = 10

# Initializing List of Ads Selected in each round
ads_selected = []
numbers_of_selections = [0] * d

# Initializing rewards of algorithm
sum_of_rewards = [0] * d
total_reward = 0

# Simulating the 10,000 Rounds as seen in the dataset
for n in range(0,N):
  # Initialize ad with the highest upper confidence bound per round
  ad = 0
  max_upper_bound = 0
  # Loop through each ad and select the ad with the highest upper confidence bound
  for i in range(0,d):
    # Initializing algorithm, each ad must be selected at least once. We will account for this in the first 10 rounds
    if numbers_of_selections[i] > 0:
      # Calculating the average reward based on the number of times the ad was selected and the sum of rewards of the ad up to the current round
      average_reward = sum_of_rewards[i] / numbers_of_selections[i]
      # Calculating the upper confidence bound
      delta_i = math.sqrt((3/2) * math.log(n+1) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    else:
      upper_bound = 1e400
    # Selects the ad with the highest upper confidence bound
    if upper_bound > max_upper_bound:
      max_upper_bound = upper_bound
      ad = i
  # List of the ad selected at each round
  ads_selected.append(ad)
  # Number of times each ad was selected
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  # Reward based on the simulated dataset, with a reward being awarded when the user clicked on the ad
  reward = dataset.values[n, ad]
  # Sum of rewards of each ad
  sum_of_rewards[ad] = sum_of_rewards[ad] + reward
  # Total reward of the algorithm
  total_reward = total_reward + reward

# Visualizing the results
plt.hist(ads_selected)
plt.title('UCB Ad Selection Distribution')
plt.xlabel('Ad Selected')
plt.ylabel('Number of Times Selected')
