import uproot
import awkward as ak
import json
import os
import statistics as st
import collections
from random import randint

#import visualkeras
from sklearn.metrics import mean_squared_error

#typical libraries
import matplotlib.pyplot as plt
import plotly
import pandas as pd
import numpy as np
import math

#tensorflow module
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#import my own functions
from functions.plot_histograms import plot_flash_time_distribution, sample_awkward_arrays, plot_variable_histograms
from functions.process_photoelectrons import process_photoelectrons
from functions.image_creator import image_creator
from functions.image_creator_visvuv_original import image_creator2
from functions.image_creator_4components import image_creator3
from functions.image_visualization import image_visualization
from functions.split_train_test import split_train_test
from functions.create_cnn_model import create_cnn_model
from functions.train_and_predict import train_and_predict
from functions.image_visualization import image_visualization