# sequential_calibration
This repository contains notebooks and python codes for sequential calibration of compartmental models, starting from manual tuning, maximum likelihood estimates and full Bayesian inference using Markov chain Monte Carlo sampling



# Folder Structure

1. manual_tuning
   - Contains codes to plot the infection data and manually tune it to find out the number of sigmoids and amplitudes
2. mle
   - Contains codes to do maximum likelihood estimation of parameters of a single and coupled PHU
3. mcmc
   - Contains codes to run parallel transitional Markov Chain Monte Carlo simulation for single and coupled PHUs.
4. PHU_Data
   - Contains the .csv files of infection data for all the 34 PHUs in Ontario from Ministry of Health, Ontario
