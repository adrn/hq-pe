# Standard library
from os import path

# Third-party
from astropy.io import fits
from astropy.table import QTable, join
import numpy as np
import yaml

# Project
from hq.config import HQ_CACHE_PATH, config_to_alldata

__all__ = ['get_metadata', 'get_ez_samples',
           'get_ms_mask', 'get_rg_mask']


def get_metadata(apply_llr_mask=True, apply_qual_mask=True):
    # Load VAC products:
    metadata = QTable.read(path.join(HQ_CACHE_PATH,
                                     'dr16/metadata-master.fits'))

    with open(path.join(HQ_CACHE_PATH, "dr16/config.yml"), "r") as f:
        config = yaml.safe_load(f.read())
    allstar, allvisit = config_to_alldata(config)

    metadata = join(metadata, allstar, keys='APOGEE_ID')
    mask = np.ones(len(metadata), dtype=bool)

    # Apply quality and log-likelihood-ratio cuts:

    if apply_llr_mask:
        llr_mask = (metadata['max_unmarginalized_ln_likelihood'] -
                    metadata['robust_constant_ln_likelihood']) > 6
        mask &= llr_mask

    if apply_qual_mask:
        qual_mask = metadata['unimodal'] | (metadata['n_visits'] >= 4)
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
