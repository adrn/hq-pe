"""Code for helping with hierarchical inference of the period-eccentricity
distribution for APOGEE binary stars.
"""
# Standard library

# Third-party
import numpy as np
from scipy.stats import truncnorm, beta
from scipy.special import logsumexp


def lnf(z, k, z0):
    return -np.log(1 + np.exp(-k * (z-z0)))


def lnalpha(z, k, z0, alpha0):
    return np.logaddexp(np.log(1-alpha0) + lnf(z, k, z0), np.log(alpha0))


def lnnormal(x, mu, var):
    return -0.5*np.log(2*np.pi) - 0.5*np.log(var) - 0.5 * (x-mu)**2 / var


def lntruncnorm(x, mu, sigma, clip_a, clip_b):
    a, b = (clip_a - mu) / sigma, (clip_b - mu) / sigma
    return truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)


class Model:

    def __init__(self, ez_nk, B1=None, B2=None, P_lim=[2, 65536]):

        # These are the fixed, assumed beta distributions we use for
        # short-period and long-period, respectively
        if B1 is None:
            B1 = beta(1.5, 50.)
        if B2 is None:
            B2 = beta(1, 1.8)

        self.ez = ez_nk  # (2, N, K)
        self.K = np.isfinite(self.ez[0]).sum(axis=-1)  # (N, )
        self.P_lim = P_lim

        # Used priors from The Joker:
        ln_e_p0 = beta.logpdf(self.ez[0], a=0.867, b=3.03)
        ln_z_p0 = np.full_like(self.ez[1],
                               -np.log(np.log(P_lim[1]) - np.log(P_lim[0])))
        self.ln_p0 = np.stack((ln_e_p0, ln_z_p0))  # (2, N, K)

        self.B1 = B1
        self.B2 = B2
        self._lnp1e = B1.logpdf(self.ez[0])
        self._lnp2e = B2.logpdf(self.ez[0])

        self._zlim = np.log(P_lim)

    @classmethod
    def unpack_pars(cls, par_arr):
        return {'lnk': par_arr[0], 'z0': par_arr[1], 'alpha0': par_arr[2],
                'muz': par_arr[3], 'lnsigz': par_arr[4]}

    def pack_pars(self, par_dict):
        return np.array([par_dict['lnk'], par_dict['z0'], par_dict['alpha0'],
                         par_dict['muz'], par_dict['lnsigz']])

    def ln_ze_dens(self, p, e, z):
        lna2 = lnalpha(z, np.exp(p['lnk']), p['z0'], p['alpha0'])
        lna1 = np.log(1 - np.exp(lna2))
        lnpe = np.logaddexp(lna1 + self.B1.logpdf(e),
                            lna2 + self.B2.logpdf(e))
        varz = np.exp(2 * p['lnsigz'])
        lnpz = lntruncnorm(z, p['muz'], varz, *self._zlim)
        return lnpe, lnpz

    def get_lnweights(self, p):
        lnpe, lnpz = self.ln_ze_dens(p, *self.ez)

        e_term = lnpe - self.ln_p0[0]
        z_term = lnpz - self.ln_p0[1]

        e_term[~np.isfinite(e_term)] = -np.inf
        z_term[~np.isfinite(z_term)] = -np.inf

        return e_term, z_term

    def ln_likelihood(self, p):
        e_lnw, z_lnw = self.get_lnweights(p)
        return logsumexp(e_lnw + z_lnw, axis=1) - np.log(self.K)

    def ln_prior(self, p):
        lp = 0.

        if not self._zlim[0] < p['z0'] < self._zlim[1]:
            return -np.inf

        if not -1 < p['lnk'] < 4:
            return -np.inf

        lp += lnnormal(p['z0'], 3.5, 1.)

        if not 1 < p['muz'] < 10:
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