# Standard library
from os import path
import pickle

# Third-party
from astropy.io import fits
from astropy.table import QTable, join
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import yaml

# Project
from hq.config import HQ_CACHE_PATH, config_to_alldata
from hq.log import logger
from model import Model

__all__ = ['get_metadata', 'get_ez_samples',
           'get_ms_mask', 'get_rg_mask']


def get_metadata(apply_qual_mask=True, min_n_visits=5):
    # Load VAC products:
    metadata = QTable.read(path.join(HQ_CACHE_PATH,
                                     'dr16/metadata-master.fits'))

    with open(path.join(HQ_CACHE_PATH, "dr16/config.yml"), "r") as f:
        config = yaml.safe_load(f.read())
    allstar, allvisit = config_to_alldata(config)

    metadata = join(metadata, allstar, keys='APOGEE_ID')
    mask = np.ones(len(metadata), dtype=bool)

    # Apply quality cuts:

    if apply_qual_mask:
        qual_mask = (metadata['unimodal'] |
                     (metadata['n_visits'] >= min_n_visits))
        qual_mask &= (metadata['LOGG'] > -0.5) & (metadata['TEFF'] > 3200)
        mask &= qual_mask

    return metadata[mask]


def get_ez_samples(apogee_ids, n_samples=256):
    samples_path = path.join(HQ_CACHE_PATH, 'dr16/samples')

    ez_samples = np.full((2, len(apogee_ids), n_samples), np.nan)
    for n, apogee_id in enumerate(apogee_ids):
        filename = path.join(samples_path, apogee_id[:4],
                             '{}.fits.gz'.format(apogee_id))
        t = fits.getdata(filename)
        K = min(n_samples, len(t))
        ez_samples[0, n, :K] = t['e'][:K]
        ez_samples[1, n, :K] = np.log(t['P'][:K])

    return ez_samples


def logg_f(teff):
    """Function used internally to separate RGB and MS"""
    slope = -0.1 / 200
    pt = (5500, 4.)

    teff = np.array(teff)
    teff_crit = 5200
    val1 = slope * (teff - pt[0]) + pt[1]
    val2 = slope * (teff_crit - pt[0]) + pt[1]

    mask = teff > teff_crit
    ret = np.zeros_like(teff)
    ret[mask] = val1[mask]
    ret[~mask] = val2
    return ret


def rg_f(teff):
    """Function used internally to separate RGB and MS"""
    slope = 0.25 / 100
    pt = (4800, 4.)
    return slope * (teff - pt[0]) + pt[1]


def get_ms_mask(teff, logg):
    ms_mask = ((logg > logg_f(teff)) &
               (logg < 5.5))
    return ms_mask


def get_rg_mask(teff, logg):
    rg_mask = ((logg <= logg_f(teff)) &
               (logg < rg_f(teff)) & (teff < 5500))
    return rg_mask


def run_pixel(name, i, ez_samples, cache_path, plot_path, pool,
              nwalkers=80):
    min_filename = path.join(cache_path, '{}_{:02d}_res.npy'.format(name, i))
    emcee_filename = path.join(cache_path,
                               '{}_{:02d}_emcee.pkl'.format(name, i))

    # Create a model instance so we can evaluate likelihood, etc.
    nparams = 5
    mod = Model(ez_samples)

    if not path.exists(min_filename) and not path.exists(emcee_filename):
        # Initial parameters for optimization
        p0 = mod.pack_pars({'lnk': 0., 'z0': np.log(30.), 'alpha0': 0.2,
                            'muz': np.log(100), 'lnsigz': np.log(4.)})

        logger.debug("{} {}: Starting minimize".format(name, i))
        res = minimize(lambda *args: -mod(*args), x0=p0, method='powell')
        min_x = res.x
        np.save(min_filename, min_x)

    # emcee run:
    logger.debug("{} {}: Done with minimize".format(name, i))

    if not path.exists(emcee_filename):
        min_x = np.load(min_filename)

        # initialization for all walkers
        all_p0 = emcee.utils.sample_ball(min_x, [1e-3] * nparams,
                                         size=nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        ndim=nparams,
                                        log_prob_fn=mod,
                                        pool=pool)
        pos, *_ = sampler.run_mcmc(all_p0, 512)
        sampler.pool = None

        with open(emcee_filename, "wb") as f:
            pickle.dump(sampler, f)

    else:
        with open(emcee_filename, "rb") as f:
            sampler = pickle.load(f)

    # Plot walker traces:
    fig, axes = plt.subplots(nparams, 1, figsize=(8, 4*nparams),
                             sharex=True)

    for k in range(nparams):
        for walker in sampler.chain[..., k]:
            axes[k].plot(walker, marker='',
                         drawstyle='steps-mid', alpha=0.4, color='k')
    axes[0].set_title(str(i))
    fig.tight_layout()
    fig.savefig(path.join(plot_path, '{}_{:02d}_trace.png'.format(name, i)),
                dpi=250)
    plt.close(fig)

    return True
