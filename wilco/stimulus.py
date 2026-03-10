import numpy as np

class StimulusBase:
    def eval(self, X, t):
        raise NotImplementedError

    def __add__(self, other):
        return _CombinedStimulus(self, other, op='+')

    def __sub__(self, other):
        return _CombinedStimulus(self, other, op='-')

    def __neg__(self):
        return _NegatedStimulus(self)


class _CombinedStimulus(StimulusBase):
    def __init__(self, a, b, op):
        self._a = a
        self._b = b
        self._op = op

    def eval(self, X, t):
        a = self._a.eval(X, t)
        b = self._b.eval(X, t)
        return a + b if self._op == '+' else a - b

class _NegatedStimulus(StimulusBase):
    def __init__(self, s):
        self._s = s

    def eval(self, X, t):
        return -self._s.eval(X, t)

class NonspecificStimulus(StimulusBase):
    def __init__(self, a):
        self.a = a
    
    def eval(self, X, t):
        return self.a

class NullStimulus(NonspecificStimulus):
    def __init__(self):
        super().__init__(a=0)

class SquareWave(StimulusBase):
    def __init__(self,
                 center, # μm
                 width, # μm
                 P, # kHz
                 duration, # ms
                 start = 0
                 ):

        self.center = center
        self.width = width
        self.P = P
        self.duration = duration
        self.start = start

    def eval(self, X, t):
        if t < self.start:
            return 0
        
        if t >= self.start + self.duration:
            return 0
        
        return self.P * (np.abs(X - self.center) <= self.width/2)
    
class PulseTrain(StimulusBase):
    def __init__(self,
                 center,    # μm
                 width,     # μm
                 P,         # kHz
                 freq,      # Hz (pulses per second)
                 duration   # ms (pulse ON time)
                 ):

        self.center = center
        self.width = width
        self.P = P
        self.freq = freq
        self.duration = duration

    def eval(self, X, t):
        period = 1000 / (self.freq)
        phase = t % period

        if phase >= self.duration:
            return 0

        return self.P * (np.abs(X - self.center) <= self.width / 2)
    
class StaticWave(StimulusBase):
    def __init__(self,
                 amplitude, # kHz
                 L, # μm
                 n, # integer
                 duration
                 ):

        self.amplitude = amplitude
        self.L = L
        self.n = n
        self.duration = duration

    def eval(self, X, t):
        if t >= self.duration:
            return 0

        return self.amplitude/2 * (1+np.cos(2*(self.n-1)*np.pi*X / self.L))
    
class FenderJuleszStimulus(StimulusBase):
    def __init__(self,
                 center, # μm
                 k, # kHz
                 v, # μm / ms
                 sigma, # μm
                 delay,
                 turnaround # μm
                 ):
        
        self.center = center
        self.k = k
        self.v = v
        self.sigma = sigma
        self.delay = delay
        self.turnaround = turnaround

    def eval(self, X, t):
        t_0 = t // self.delay

        d = self.v*t_0

        if 2*d < self.turnaround:
            pass
        elif 2*d < 2*self.turnaround:
            d = d - self.turnaround
        else:
            d = 0

        return self.k * (np.exp(-(X - self.center - d)**2/self.sigma**2) + np.exp(-(X - self.center + d)**2/self.sigma**2))