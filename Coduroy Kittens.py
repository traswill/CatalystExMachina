# Because why the hell not

from sklearn.ensemble import RandomForestRegressor

class catalyst():
    def __init__(self):
        self.labels = None
        self.features = None

    def set(self, labels=None, features=None):
        self.labels = labels
        self.features = features




