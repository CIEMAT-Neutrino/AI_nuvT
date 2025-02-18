import uproot
import awkward as ak
import json
import os
import statistics as st
import collections
from random import randint
import pyarrow as pa
import pyarrow.parquet as pq
import visualkeras

#import visualkeras
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
from functions.Data_analysis import plot_flash_time_distribution, sample_awkward_arrays, plot_variable_histograms
from functions._2_PE_time_matrices import process_photoelectrons, process_photoelectrons2
from functions._4_Image_creation_visualization import plot_image, plot_image2, image_creator_gen, image_creator_gen_inv, image_creator_gen_2images, image_creator_gen_2images_inv, alberto_image, image_creator_gen_inv_nuevo, image_creator_gen_inv_nuevo2
from functions._5_Regression_AI import split_train_test, train_and_predict

