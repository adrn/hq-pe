"""Infer population parameters along the main sequence, in bins of TEFF"""

# Standard library
import os
from os import path
import pickle
import sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import emcee
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIPool

# Project
from hq.log import logger
from hq.script_helpers import get_parser

from scipy.optimize import minimize


from model import Model
from helpers import get_metadata, get_ez_samples, get_ms_mask

scripts_path = path.split(path.abspath(__file__))[0]
cache_path = path.abspath(path.join(scripts_path, '../cache/'))
plot_path = path.abspath(path.join(scripts_path, '../plots/'))


def main(pool, overwrite=False):
    # Config stuff:
    teff_step = 300
    teff_binsize = 1.5 * teff_step
    teff_bincenters = np.arange(3400, 7000+1e-3, teff_step)

    # emcee config:
    nparams = 5
    nwalkers = 80

    # Load all data:
    metadata = get_metadata()
    ms_mask = get_ms_mask(metadata['TEFF'], metadata['LOGG'])
    metadata = metadata[ms_mask]

    # Make sure paths exist:
    for path_ in [cache_path, plot_path]:
        os.makedirs(path_, exist_ok=True)

    for i, ctr in enumerate(teff_bincenters):
        min_filename = path.join(cache_path, 'ms_{:02d}_res.npy')
        emcee_filename = path.join(cache_path, 'ms_{:02d}_emcee.pkl')

        l = ctr - teff_binsize / 2
        r = ctr + teff_binsize / 2
        pixel_mask = ((metadata['TEFF'] > l) & (metadata['TEFF'] <= r))

        # Load samples for this bin:
        logger.debug("{}: Loading samples".format(i))
        ez_samples = get_ez_samples(metadata['APOGEE_ID'][pixel_mask])

        # Create a model instance so we can evaluate likelihood, etc.
        mod = Model(ez_samples)

        if not path.exists(min_filename) and not path.exists(emcee_filename):
            # Initial parameters for optimization
            p0 = mod.pack_pars({'lnk': 0., 'z0': np.log(30.), 'alpha0': 0.2,
                                'muz': np.log(100), 'lnsigz': np.log(4.)})

            logger.debug("{}: Starting minimize".format(i))
            res = minimize(lambda *args: -mod(*args), x0=p0, method='powell')
            min_x = res.x
            np.save(min_filename, min_x)

        else:
            min_x = np.load(min_filename)

        # emcee run:
        logger.debug("{}: Done with minimize")

        if not path.exists(emcee_filename):
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

        for k in range(sampler.chain.shape[-1]):
            for walker in sampler.chain[..., k]:
                axes[k].plot(walker, marker='',
                             drawstyle='steps-mid', alpha=0.4, color='k')
        axes[0].set_title('MS {}'.format(i))
        fig.tight_layout()
        fig.savefig(path.join(plot_path, 'ms_{:02d}_trace.png'), dpi=250)
        plt.close(fig)


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Hierarchical inference for main sequence',
                        loggers=logger)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite stuff.")

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(pool=pool, overwrite=args.overwrite)

    sys.exit(0)
