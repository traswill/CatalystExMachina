# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Code Start: December 7, 2018

from TheKesselRun.Code.Plotter import Graphic
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

class BayesianLearner():
    def __init__(self):

        # Data_space is the current known dataset in the form X_ds
        self.data_space = None

        # Parameter_space is the experimental design that is to be explored, in the form X_ps
        self.parameter_space = None

