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

def check_params(params, epsilon = .1):
    # refractory period must be much smaller than membrane time constant
    assert params["r_E"] / params["tau"] <= epsilon
    assert params["r_I"] / params["tau"] <= epsilon

    # resting state E=I=0 must be stable to small perturbations
    assert params["rho_E"] * params["v_E"] * params["b_EE"] * params["sigma_EE"] / params["theta_E"] <= epsilon
    assert params["rho_E"] * params["v_I"] * params["b_II"] * params["sigma_II"] / params["theta_I"] <= epsilon

    # uniformly excited state should be unstable in absence of stimulus
    assert 1/(1 + params["r_E"]) + 2*params["b_EE"]*params["sigma_EE"] - 2*params["b_IE"]*params["sigma_IE"]/(1+params["r_I"]) < params["theta_E"]
    assert 1/(1 + params["r_E"]) + 2*params["b_EI"]*params["sigma_EI"] - 2*params["b_II"]*params["sigma_II"]/(1+params["r_I"]) > params["theta_I"]
 
    # E → I interaction must be longer range than E → E
    assert params["sigma_EI"] > params["sigma_EE"]

class WilsonCowan:
    def __init__(self, **kwargs):
        param_names = {'tau', 'alpha', 'r_E', 'r_I', 'rho_E', 'rho_I', 'v_E', 'v_I', 'theta_E', 'theta_I', 'b_EE', 'b_IE', 'b_EI', 'b_II', 'sigma_EE', 'sigma_IE', 'sigma_EI', 'sigma_II'}

        if set(kwargs) != param_names:
            raise ValueError(f"Expected kwargs [{', '.join(param_names)}], got [{', '.join(list(kwargs))}]")
        
        check_params(kwargs)

        for param, val in kwargs.items():
            setattr(self, param, val)

        for k in ("E", "I"):
            v = kwargs[f"v_{k}"]
            theta = kwargs[f"theta_{k}"]
            S = lambda n, v=v, theta=theta: 1/(1 + np.exp(-v*(n-theta))) - 1/(1 + np.exp(v*theta))
            setattr(self, f"S_{k}", S) # mV → adim

        for k in ("E", "I"):
            for l in ("E", "I"):
                b = kwargs[f"b_{k}{l}"]
                sigma = kwargs[f"sigma_{k}{l}"]
                beta = lambda x, b=b, sigma=sigma: b*np.exp(-np.abs(x)/sigma)
                setattr(self, f"beta_{k}{l}", beta)

    def run(self, xmin, xmax, num_nodes, num_steps=None, t_max=None, dt=1, P=NullStimulus(), Q=NullStimulus(), dtype="float32"):
        if num_steps is None:
            assert t_max is not None
            num_steps = int(t_max / dt)

        dx = (xmax - xmin) / (num_nodes - 1)
        X = np.linspace(xmin, xmax, num_nodes, dtype=dtype)
        T = np.linspace(0, t_max, num_steps, dtype=dtype)

        def conv(field, beta_fn):
            K = beta_fn(X - X[0])
            K = np.concatenate([K[:0:-1], K])

            return dx * np.convolve(field, K, mode="full")[num_nodes-1:2*num_nodes-1]

        E = np.zeros((num_steps, num_nodes), dtype=dtype)
        I = np.zeros((num_steps, num_nodes), dtype=dtype)

        p = np.zeros((num_steps, num_nodes))
        q = np.zeros((num_steps, num_nodes))

        for i in tqdm(range(num_steps - 1)):
            conv_ee = conv(E[i], self.beta_EE)
            conv_ie = conv(I[i], self.beta_IE)
            conv_ei = conv(E[i], self.beta_EI)
            conv_ii = conv(I[i], self.beta_II)

            p[i] = P.eval(X, i*dt)
            q[i] = Q.eval(X, i*dt)

            Ne = self.alpha * (conv_ee - conv_ie + p[i])
            Ni = self.alpha * (conv_ei - conv_ii + q[i])

            Se = self.S_E(Ne)
            Si = self.S_I(Ni)

            dE = (-E[i] + (1.0 - self.r_E * E[i]) * Se) / self.tau
            dI = (-I[i] + (1.0 - self.r_I * I[i]) * Si) / self.tau

            E[i+1] = E[i] + dt * dE
            I[i+1] = I[i] + dt * dI

        return ActivityCurves(E, I, X, T, p, q)

class ActiveTransientModel(WilsonCowan):
    def __init__(self):
        super().__init__(
            tau = 10,
            alpha = 1,

            r_E = 1,
            r_I = 1,

            rho_E = 1, # mm⁻¹
            rho_I = 1, # mm⁻¹

            v_E = 0.5, # mV
            v_I = 0.3,

            theta_E = 9.0, # mV⁻¹
            theta_I = 17.0,

            b_EE = 1.5, # adim
            b_IE = 1.35,
            b_EI = 1.35,
            b_II = 1.8,

            sigma_EE = 40, # mm
            sigma_IE = 60, # mm
            sigma_EI = 60, # mm
            sigma_II = 30, # mm
        )

class OscillatoryModel(WilsonCowan):
    def __init__(self):
        super().__init__(
            tau = 10, # ms
            alpha = 1, # mV

            r_E = 1, # ms
            r_I = 1, # ms

            rho_E = 1, # mm⁻¹
            rho_I = 1, # mm⁻¹

            v_E = 0.5, # mV
            v_I = 1,

            theta_E = 9.0, # mV⁻¹
            theta_I = 15.0,

            b_EE = 2.0, # adim
            b_IE = 1.5,
            b_EI = 1.5,
            b_II = 0.1,

            sigma_EE = 40, # mm
            sigma_IE = 60, # mm
            sigma_EI = 60, # mm
            sigma_II = 20, # mm
        )

class SteadyStateModel(WilsonCowan):
    def __init__(self):
        super().__init__(
            tau = 10, # ms
            alpha = 1, # mV

            r_E = 1, # ms
            r_I = 1, # ms

            rho_E = 1, # mm⁻¹
            rho_I = 1, # mm⁻¹

            v_E = 0.5, # mV
            v_I = 0.3,

            theta_E = 9.0, # mV⁻¹
            theta_I = 17.0,

            b_EE = 2.0, # adim
            b_IE = 1.35,
            b_EI = 1.35,
            b_II = 1.8,

            sigma_EE = 40, # mm
            sigma_IE = 60, # mm
            sigma_EI = 60, # mm
            sigma_II = 30, # mm
        )