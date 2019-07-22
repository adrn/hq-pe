"""Infer population parameters along the main sequence, in bins of TEFF"""

# Standard library
import os
from os import path
import sys

# Third-party
import numpy as np
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIPool

# Project
from hq.log import logger
from hq.script_helpers import get_parser

from helpers import get_metadata, get_ez_samples, get_ms_mask, run_pixel

scripts_path = path.split(path.abspath(__file__))[0]
cache_path = path.abspath(path.join(scripts_path, '../cache/'))
plot_path = path.abspath(path.join(scripts_path, '../plots/'))


def main(pool, overwrite=False):
    # Config stuff:
    name = 'ms'
    teff_step = 300
    teff_binsize = 1.5 * teff_step
    teff_bincenters = np.arange(3400, 7000+1e-3, teff_step)

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

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(pool=pool, overwrite=args.overwrite)

    sys.exit(0)
