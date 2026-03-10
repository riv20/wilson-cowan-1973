from tqdm import tqdm

import pandas as pd
import numpy as np

from .stimulus import NullStimulus

class ActivityCurves:
    def __init__(self, E, I, X, T, P, Q):
        self.E = E
        self.I = I
        self.X = X
        self.T = T
        self.P = P
        self.Q = Q

    def long_form(self):
        num_steps, num_nodes = self.E.shape
        res = pd.DataFrame({
                "t": np.repeat(self.T, num_nodes),
                "x": np.tile(self.X, num_steps),
                "E": self.E.ravel(),
                "I": self.I.ravel(),
                "P": self.P.ravel(),
                "Q": self.Q.ravel()
            })

        res.index = list(zip(np.repeat(np.arange(num_steps), num_nodes), np.tile(np.arange(num_nodes), num_steps)))

        return res
    
    def get(self, t = None, x = None):
        if x is not None:
            idx_x = np.argmin(np.abs(self.X - x))

            if x > self.X.max() or x < self.X.min():
                raise ValueError("x not in range")
            
        if t is not None:
            idx_t = np.argmin(np.abs(self.T - t))

            if t > self.T.max() or t < self.T.min():
                raise ValueError("t not in range")
            
        if t is None and x is None:
            return None
        
        if t is not None and x is None:
            return pd.DataFrame(dict(x=self.X, E=self.E[idx_t], I = self.I[idx_t], P = self.P[idx_t], Q = self.Q[idx_t]))
        
        if t is None and x is not None:
            return pd.DataFrame(dict(t=self.T, E=self.E[:, idx_x], I = self.I[:, idx_x], P = self.P[:, idx_x], Q = self.Q[:, idx_x]))

        if t is not None and x is not None:
            return pd.DataFrame(dict(E = [self.E[idx_t, idx_x]], I = [self.I[idx_t, idx_x]], P = [self.P[idx_t, idx_x]], Q = [self.Q[idx_t, idx_x]]))

class WilsonCowan:
    def __init__(self, **kwargs):
        param_names = {'mu', # ms
                       'alpha', # adim

                       'r_E', # ms
                       'r_I',

                       'F_E', # kHz
                       'F_I',

                       'rho_E', # μm⁻¹
                       'rho_I',

                       'v_E', # adim
                       'v_I',

                       'theta_E', # adim
                       'theta_I',

                       'b_EE', # adim
                       'b_IE',
                       'b_EI',
                       'b_II',

                       'sigma_EE', # μm
                       'sigma_IE',
                       'sigma_EI',
                       'sigma_II'
                       }

        if set(kwargs) != param_names:
            raise ValueError(f"Expected kwargs [{', '.join(param_names)}], got [{', '.join(list(kwargs))}]")
        
        for param, val in kwargs.items():
            setattr(self, param, val)

        for k in ("E", "I"):
            v = kwargs[f"v_{k}"]
            theta = kwargs[f"theta_{k}"]
            S = lambda n, v=v, theta=theta: 1/(1 + np.exp(-v*(n-theta))) - 1/(1 + np.exp(v*theta))
            setattr(self, f"S_{k}", S) # int → adim

        for k in ("E", "I"):
            for l in ("E", "I"):
                b = kwargs[f"b_{k}{l}"]
                sigma = kwargs[f"sigma_{k}{l}"]
                beta = lambda x, b=b, sigma=sigma: b*np.exp(-np.abs(x)/sigma) # μm → adim
                setattr(self, f"beta_{k}{l}", beta)

    def run(self,
            
            xmin, # μm
            xmax, # μm

            num_nodes, # int
            num_steps=None, # int
            
            t_max=None, # ms
            dt=1, # ms

            P=NullStimulus(), # (ms, μm) → kHz
            Q=NullStimulus(), # (ms, μm) → kHz

            dtype="float32"
            ):
        
        if num_steps is None:
            assert t_max is not None
            num_steps = int(t_max / dt)

        dx = (xmax - xmin) / (num_nodes - 1) # μm
        X = np.linspace(xmin, xmax, num_nodes, dtype=dtype) # μm
        T = np.linspace(0, t_max, num_steps, dtype=dtype) # ms

        def conv(field, beta_fn):
            K = beta_fn(X - X[0])
            K = np.concatenate([K[:0:-1], K]) # μm

            return dx * np.convolve(field, K, mode="full")[num_nodes-1:2*num_nodes-1]

        E = np.zeros((num_steps, num_nodes), dtype=dtype) # kHz
        I = np.zeros((num_steps, num_nodes), dtype=dtype) # kHz

        p = np.zeros((num_steps, num_nodes))
        q = np.zeros((num_steps, num_nodes))

        for i in tqdm(range(num_steps - 1)):
            conv_ee = conv(E[i], self.beta_EE) # kHz × μm = mm/s
            conv_ie = conv(I[i], self.beta_IE)
            conv_ei = conv(E[i], self.beta_EI)
            conv_ii = conv(I[i], self.beta_II)

            p[i] = P.eval(X, i*dt) # kHz
            q[i] = Q.eval(X, i*dt)

            # [rho × conv] = kHz
            Ne = self.alpha*self.mu*(self.rho_E*conv_ee - self.rho_I*conv_ie + p[i]) # adim
            Ni = self.alpha*self.mu*(self.rho_E*conv_ei - self.rho_I*conv_ii + q[i])

            Se = self.S_E(Ne) # adim
            Si = self.S_I(Ni)

            dE = (-E[i] + self.F_E*(1.0 - self.r_E*E[i])*Se) / self.mu # kHz / ms
            dI = (-I[i] + self.F_I*(1.0 - self.r_I*I[i])*Si) / self.mu

            E[i+1] = E[i] + dt*dE
            I[i+1] = I[i] + dt*dI

        return ActivityCurves(E, # kHz
                              I, # kHz
                              X, # μm
                              T, # ms
                              p, # kHz
                              q, # kHz
                              )

class ActiveTransientModel(WilsonCowan):
    def __init__(self):
        super().__init__(
            mu = 10,
            alpha = .1,

            r_E = 1,
            r_I = 1,

            F_E = 1,
            F_I = 1,

            rho_E = 1,
            rho_I = 1,

            v_E = 0.5,
            v_I = 0.3,

            theta_E = 9.0,
            theta_I = 17.0,

            b_EE = 1.5,
            b_IE = 1.35,
            b_EI = 1.35,
            b_II = 1.8,

            sigma_EE = 40,
            sigma_IE = 60,
            sigma_EI = 60,
            sigma_II = 30,
        )

class OscillatoryModel(WilsonCowan):
    def __init__(self):
        super().__init__(
            mu = 10,
            alpha = .1,

            r_E = 1,
            r_I = 1,

            F_E = 1,
            F_I = 1,

            rho_E = 1,
            rho_I = 1,

            v_E = 0.5,
            v_I = 1,

            theta_E = 9.0,
            theta_I = 15.0,

            b_EE = 2.0,
            b_IE = 1.5,
            b_EI = 1.5,
            b_II = 0.1,

            sigma_EE = 40,
            sigma_IE = 60,
            sigma_EI = 60,
            sigma_II = 20,
        )

class SteadyStateModel(WilsonCowan):
    def __init__(self):
        super().__init__(
            mu = 10,
            alpha = .1,

            r_E = 1,
            r_I = 1,

            F_E = 1,
            F_I = 1,

            rho_E = 1,
            rho_I = 1,

            v_E = 0.5,
            v_I = 0.3,

            theta_E = 9.0,
            theta_I = 17.0,

            b_EE = 2.0,
            b_IE = 1.35,
            b_EI = 1.35,
            b_II = 1.8,

            sigma_EE = 40,
            sigma_IE = 60,
            sigma_EI = 60,
            sigma_II = 30,
        )