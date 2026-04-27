# Automated Classification of Intermittent Pulsars Utilising Convolutional Neural Networks

**Author:** Daniel José Barranco Aragón  
**Institution:** University of East Anglia (School of Engineering, Mathematics, and Physics)  
**Year:** 2026  

## Project Overview
This repository contains the complete, executable codebase for my BSc dissertation. The project addresses the classification bottleneck associated with intermittent pulsars in radio astronomy. Traditional amplitude-based matched filters frequently fail when processing faint, highly variable sources heavily distorted by interstellar scintillation. 

To resolve this, the codebase implements a supervised deep learning pipeline. It utilises a one-dimensional Convolutional Neural Network (CNN) to classify radio signals strictly by their geometric pulse morphology rather than absolute flux density.

## Repository Structure
* `/data_processing/`: Scripts used for dimensionality reduction (frequency scrunching) and min-max normalisation.
* `/models/`: The Keras/TensorFlow definitions for the 1D Convolutional Neural Network.
* `/notebooks/`: Annotated Jupyter Notebooks demonstrating the model training dynamics, dataset balancing (inverse-frequency class weighting), and testing evaluations.

## Core Dependencies
Executed environments require the following primary libraries:
* `Python 3.x`
* `TensorFlow / Keras`
* `SciPy` (for Fourier-based resampling)
* `NumPy` & `Pandas`
* `PyPulse` (for raw pulsar archive manipulation)

## Citation
If you utilise this code or methodology in your own research, please cite the repository as follows:
> Barranco Aragón, D. J. (2026). *Automated Classification of Intermittent Pulsars Utilising Convolutional Neural Networks*. GitHub. https://github.com/DanielBarranco-Phy/automated_classification_of_intermittent_pulsars_utilising_convolutional_neural_networks
