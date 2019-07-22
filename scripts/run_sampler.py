"""Infer population parameters along the red giant branch, in bins of LOGG"""

# Standard library
import os
from os import path
import sys

# Third-party
import numpy as np
from schwimmbad import choose_pool

# Project
from hq.log import logger
from hq.script_helpers import get_parser

from helpers import (get_metadata, get_ez_samples, get_rg_mask, get_ms_mask,
                     run_pixel)

scripts_path = path.split(path.abspath(__file__))[0]
cache_path = path.abspath(path.join(scripts_path, '../cache/'))
plot_path = path.abspath(path.join(scripts_path, '../plots/'))

logg_step = 0.25
logg_binsize = 1.5 * logg_step
logg_bincenters = np.arange(0, 4+1e-3, logg_step)

teff_step = 300
teff_binsize = 1.5 * teff_step
teff_bincenters = np.arange(3400, 7000+1e-3, teff_step)


def main_rg(pool, overwrite=False):
    # Config stuff:
    name = 'rg'

    # Load all data:
    metadata = get_metadata()
    rg_mask = get_rg_mask(metadata['TEFF'], metadata['LOGG'])
    metadata = metadata[rg_mask]

    # Make sure paths exist:
    for path_ in [cache_path, plot_path]:
        os.makedirs(path_, exist_ok=True)

    for i, ctr in enumerate(logg_bincenters):
        l = ctr - logg_binsize / 2
        r = ctr + logg_binsize / 2
        pixel_mask = ((metadata['LOGG'] > l) & (metadata['LOGG'] <= r))

        # Load samples for this bin:
        logger.debug("{} {}: Loading samples".format(name, i))
        ez_samples = get_ez_samples(metadata['APOGEE_ID'][pixel_mask])

        # Run
        run_pixel(name, i, ez_samples, cache_path, plot_path, pool,
                  nwalkers=80)


def main_ms(pool, overwrite=False):
    # Config stuff:
    name = 'ms'

    # Load all data:
    metadata = get_metadata()
    ms_mask = get_ms_mask(metadata['TEFF'], metadata['LOGG'])
    metadata = metadata[ms_mask]

    # Make sure paths exist:
    for path_ in [cache_path, plot_path]:
        os.makedirs(path_, exist_ok=True)

    for i, ctr in enumerate(teff_bincenters):
        l = ctr - teff_binsize / 2
        r = ctr + teff_binsize / 2
        pixel_mask = ((metadata['TEFF'] > l) & (metadata['TEFF'] <= r))

        # Load samples for this bin:
        logger.debug("{} {}: Loading samples".format(name, i))
        ez_samples = get_ez_samples(metadata['APOGEE_ID'][pixel_mask])

        # Run
        run_pixel(name, i, ez_samples, cache_path, plot_path, pool,
                  nwalkers=80)


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

    parser.add_argument("-n", "--name", dest="name", required=True, type=str,
                        help="Name; must be either 'ms' or 'rg'")

    args = parser.parse_args()

    pool = choose_pool(mpi=args.mpi)
    try:
        if args.name == 'ms':
            main_ms(pool=pool, overwrite=args.overwrite)

        elif args.name == 'rg':
            main_rg(pool=pool, overwrite=args.overwrite)

        else:
            logger.error("Invalid name '{}'".format(args.name))

        pool.close()
    except Exception as e:
        pool.close()
        raise e

    sys.exit(0)
