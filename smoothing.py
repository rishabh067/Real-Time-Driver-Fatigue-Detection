# smoothing.py

class ExponentialSmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev = None

    def smooth(self, value):
        if self.prev is None:
            self.prev = value
        else:
            self.prev = self.alpha * self.prev + (1 - self.alpha) * value
        return self.prev
