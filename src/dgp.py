import numpy as np
from scipy.stats import random_correlation
from numpy.random import default_rng

class SimulatedDataset:

    '''
    description
    '''
    def __init__(self, N,  d, alpha, seed=123):
        self.rng = default_rng(seed)
        self.N = N
        self.d = d
        self.alpha = alpha

        self.X = self.generate_features()
        self.x_confounders_idx, self.x_prognostic_idx,  self.x_effect_mod_idx = self.set_covariate_types()

        self.mu0 = self.set_mu0()
        self.tau = self.set_tau()
        self.mu1 = self.set_mu1()
        self.e = self.set_propensity_fn()
        self.Y0, self.Y1 = self.generate_potential_outcomes()

        self.W = self.generate_treatment()
        self.Y = self.generate_outcome()
        
    # --------------------------
    # CORRELATION STRUCTURE

    def generate_random_eigenvector(self):
        eigvals = self.rng.dirichlet(np.ones(self.d)) * self.d
        return eigvals

    def generate_random_correlation_matrix(self):
        eigvals = self.generate_random_eigenvector()
        corr_matrix = random_correlation.rvs(eigvals, random_state=self.rng)
        return corr_matrix

    def generate_features(self):
        corr_matrix = self.generate_random_correlation_matrix()
        X = self.rng.multivariate_normal(mean=np.zeros(self.d), cov=corr_matrix, size=self.N)
        return X
    
    def set_covariate_types(self):
        x_confounders_idx = [0, 1, 2, 3]
        x_prognostic_idx = [4, 5, 6]
        x_effect_mod_idx = [7, 8, 9]
        return x_confounders_idx, x_prognostic_idx, x_effect_mod_idx

    # --------------------------
    # STRUCTURAL FUNCTIONS

    def rho(self, x):
        # example heterogeneity function
        return 2 / (1 + np.exp(-5 * (x - 0.35)))

    def set_mu0(self):
        # function of confounding and prognostic covariates
        Xc = self.X[:, self.x_confounders_idx]
        Xp = self.X[:, self.x_prognostic_idx]

        mu0 = (
            0.6 * Xc[:, 0]
            - 0.3 * Xc[:, 1]
            + 0.2 * Xc[:, 2]
            - 0.1 * Xc[:, 3]**2
            + 0.5 * Xp[:, 0]
            - 0.2 * Xp[:, 1]
            + 0.1 * Xp[:, 2]**2
        )
        return mu0
    
    def set_tau(self):
        # function of effect modifier covariates
        Xem = self.X[:, self.x_effect_mod_idx]
        tau = (
            0.8 * self.rho(Xem[:, 0])
            - 0.4 * self.rho(Xem[:, 1])
            + 0.2 * self.rho(Xem[:, 2]**2)
        )
        return tau
        

    def set_mu1(self):
        # mu1 = mu0 + tau
        return self.mu0 + self.set_tau()

    def generate_potential_outcomes(self):
        eps = self.rng.normal(0, 1, size=self.N)
        Y0 = self.mu0 + eps
        Y1 = self.mu1 + eps
        return Y0, Y1

    def set_propensity_fn(self):
        '''
        Propensity score function determining probability of treatment = 1
            e(x) = P(W=1|X=x),
        with 'alpha' controlling strength of confounding
        (alpha = 0 is completely random treatment assignment)
        '''
        Xc = self.X[:, self.x_confounders_idx]
        logits = (
            0.6 * Xc[:, 0]
            - 0.4 * Xc[:, 1]
            + 0.2 * Xc[:, 2]
            - 0.1 * Xc[:, 3]**2
        )
        return 1 / (1 + np.exp(-logits*self.alpha + 0.3))

    # -------------------------
    # OUTCOME AND TREATMENT

    def generate_treatment(self):
        return self.rng.binomial(1, self.e)

    def generate_outcome(self):
        return np.where(self.W == 1, self.Y1, self.Y0)
