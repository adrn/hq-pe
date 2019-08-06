"""Code for helping with hierarchical inference of the period
distribution for APOGEE binary stars.
"""
# Standard library

# Third-party
import numpy as np
from scipy.stats import truncnorm
from scipy.special import logsumexp


def lntruncnorm(x, mu, sigma, clip_a, clip_b):
    a, b = (clip_a - mu) / sigma, (clip_b - mu) / sigma
    return truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)


class Model:

    def __init__(self, z_nk, P_lim=[2, 65536]):
        """z = log10(P)"""

        self.z = z_nk  # (N, K)
        self.K = np.isfinite(self.z).sum(axis=-1)  # (N, )
        self.P_lim = P_lim

        # Used priors from The Joker:
        ln_z_p0 = np.full_like(self.z,
                               -np.log(np.log(P_lim[1]) - np.log(P_lim[0])))
        self.ln_p0 = ln_z_p0  # (N, K)
        self._zlim = np.log10(P_lim)

    @classmethod
    def unpack_pars(cls, par_arr):
        return {'muz': par_arr[0], 'lnsigz': par_arr[1]}

    def pack_pars(self, par_dict):
        return np.array([par_dict['muz'], par_dict['lnsigz']])

    def get_lnweights(self, p):
        lnpe, lnpz = self.ln_ze_dens(p, *self.ez)

        e_term = lnpe - self.ln_p0[0]
        z_term = lnpz - self.ln_p0[1]

        e_term[~np.isfinite(e_term)] = -np.inf
        z_term[~np.isfinite(z_term)] = -np.inf

        return e_term, z_term

    def ln_likelihood(self, p):
        varz = np.exp(p['lnsigz']) ** 2
        lnpz = lntruncnorm(self.z, p['muz'], varz, *self._zlim)
        lnweight = lnpz - self.ln_p0
        lnweight[~np.isfinite(lnweight)] = -np.inf
        return logsumexp(lnweight, axis=1) - np.log(self.K)

    def ln_prior(self, p):
        lp = 0.

        if not 1 < p['muz'] < 10:
            return -np.inf

        if not -1 < p['lnsigz'] < 10:
            return -np.inf

        return lp

    def ln_prob(self, par_arr):
        p = self.unpack_pars(par_arr)

        lp = self.ln_prior(p)
        if not np.isfinite(lp):
            return -np.inf

        ll_n = self.ln_likelihood(p)
        if not np.all(np.isfinite(ll_n)):
            return -np.inf

        return np.sum(ll_n)

    def neg_ln_prob(self, *args, **kwargs):
        return -self.ln_prob(*args, **kwargs)

    def __call__(self, p):
        return self.ln_prob(p)
