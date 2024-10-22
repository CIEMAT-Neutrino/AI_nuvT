# AI_nuvT
Artificial Intelligence code to predict the neutrino interaction time in a LArTPC experiment

## Goal:
The goal of this project is to develop a neural network that uses the scintillation light produced in SBND, a liquid argon TPC exposed to a neutrino beam at Fermilab. The neural network uses the number of photoelectrons observed in each of the channels in an event as input. The output estimate is the temporal coordinate of the neutrino interaction.

## Objectives: 
1. Suppress cosmics. Neutrinos should come in packets (the proton beam from which they are created sends packets of protons) while cosmics (mainly muons) have a random distribution. So if you know how to match the detector light with the trace and the interaction time you can differentiate neutrino events from noise.   
2. Search for BSM phenomena. Search between neutrino packets for packet tails that should not appear. Search if they agree with m-LLP (massive long live particles) or HNLs modeling.

## Implementation:
1. Import data (in OpDetAnalyzer format).
2. Construct PE & time matrices.
3. Construct maps (representations of the detector)
3. Use the maps + PE & time matrices to construct image (input). It contains information of the detectors about photoelectrons collected and weighted times measured. 
4. Use a CNN (convolutional layers + FCNN) to make predictions of the regression problem.

# Author
Sergio Domínguez Vidales
