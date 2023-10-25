class TimeSeries:
    def __init__(self, start, end, intervalLength):
        self.intervals = [TimeInterval(start, start + intervalLength - 1) for start in range(start, end, intervalLength)]

class TimeInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end