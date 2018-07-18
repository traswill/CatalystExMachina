# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

import pandas as pd
import numpy as np

class Observation():
    def __init__(self):
        self.temperature = None
        self.pressure = None
        self.space_velocity = None
        self.gas = None
        self.concentration = None
        self.reactor = None

        self.activity = None
        self.selectivity = None


