"""
Shared pieces of the movie-track pipeline.

- MASTER_*: where the one-off master histories live and which SOAP fields go in.
- match_track_ids(): locate TrackIds in one snapshot's catalogue.
- finalise_histories(): turn raw M*/SFR/n_bound histories into the arrays the
  frame script reads (log values, quenched floor, orphan flag). Used by
  select_tracks_for_movie.py.

Every script here imports this module first: it puts simulation_envr/code on
sys.path so SWIFT_tools, Tweb_tools and plotting_funcs can be imported.
"""

import sys

import numpy as np

ROOT = '/cosma8/data/dp004/dc-howa2/colibre_movies'
ENVR_CODE = '/cosma8/data/dp004/dc-howa2/simulation_envr/code'
if ENVR_CODE not in sys.path:
    sys.path.insert(0, ENVR_CODE)

from SWIFT_tools import to_log_mass, to_log_SFR  # noqa: E402

DATA_DIR = f'{ROOT}/data'
FRAMES_DIR = f'{ROOT}/frames'
ANIM_DIR = f'{ROOT}/animations'
LOG_DIR = f'{ROOT}/logs'
MASTER_DIR = f'{DATA_DIR}/master_tracks'
MASTER_FILE = f'{MASTER_DIR}/track_histories_logM8.hdf5'
MASTER_IDS = f'{MASTER_DIR}/track_ids_z0.npz'
MASTER_PARTS = f'{MASTER_DIR}/parts'

SIM = dict(simulation='colibre', run='L400_m7', mode='Thermal')
APERTURE = 'exclusive_sphere_50kpc'
H = 0.681
Z_SUN = 0.0134  # solar metal mass fraction, for log10(Z/Zsun)

# Bottom of the sSFR axis in the animation frames, in log10(Gyr^-1).
# Quenched galaxies (SFR == 0) are pinned here so they stay visible.
SSFR_FLOOR = -4.9

# Below this bound-particle count an HBT+ track is treated as an orphan
# remnant rather than a resolved subhalo.
NBOUND_MIN = 20

# Per-snapshot float fields stored in the master file:
#   name in file -> (SOAP group, SOAP field, column of a 2D field or None)
MASTER_FLOAT_FIELDS = {
    'stellar_mass':           (APERTURE, 'stellar_mass', None),
    'star_formation_rate':    (APERTURE, 'star_formation_rate', None),
    'gas_mass':               (APERTURE, 'gas_mass', None),
    'half_mass_radius_stars': (APERTURE, 'half_mass_radius_stars', None),
    'stellar_metallicity':    (APERTURE, 'stellar_mass_fraction_in_metals', None),
    'gas_metallicity':        (APERTURE, 'gas_mass_fraction_in_metals', None),
    'lum_g':                  (APERTURE, 'stellar_luminosity', 1),
    'lum_r':                  (APERTURE, 'stellar_luminosity', 2),
    # host_m200c is filled separately (needs the host gather)
}
MASTER_SNAP_FIELDS = list(MASTER_FLOAT_FIELDS) + ['host_m200c', 'n_bound',
                                                  'is_central', 'present']


def match_track_ids(track_id_arr, track_ids):
    """
    Find each of `track_ids` in one snapshot's TrackId array.

    Returns (found, rows): `found` is a bool mask over `track_ids`, and
    `rows` are the catalogue rows of the found ones, in the same order.
    """
    if np.unique(track_id_arr).size != track_id_arr.size:
        raise ValueError("Duplicate TrackIds in snapshot catalogue")
    sorter = np.argsort(track_id_arr)
    pos = np.searchsorted(track_id_arr, track_ids, sorter=sorter)
    pos = np.clip(pos, 0, len(track_id_arr) - 1)
    found = track_id_arr[sorter][pos] == track_ids
    rows = sorter[pos[found]]
    return found, rows


def finalise_histories(mstar, sfr, nbound, present):
    """
    Convert raw (n_snaps, n_tracks) histories to the frame-script arrays.

    mstar, sfr : SOAP units (1e10 Msun, SOAP SFR units), NaN where absent
    nbound     : bound particle count, -1 where absent
    present    : bool, matched in the catalogue

    Returns dict with log_mstar, log_sfr, log_ssfr (clipped at SSFR_FLOOR,
    quenched pinned to it), quenched, orphan.
    """
    mstar = np.asarray(mstar, dtype=np.float64)
    sfr = np.asarray(sfr, dtype=np.float64)

    # A track can be matched but have SFR == 0 (quenched) or M* == 0 (an
    # orphan remnant with no bound stars). Blank those before taking logs,
    # then restore the quenched ones at the floor.
    quenched = present & (sfr == 0.0)
    no_stars = present & (mstar == 0.0)

    mstar_safe = np.where(mstar > 0.0, mstar, np.nan)
    sfr_safe = np.where(sfr > 0.0, sfr, np.nan)

    log_mstar = to_log_mass(mstar_safe)
    log_sfr = to_log_SFR(sfr_safe)

    # sSFR in Gyr^-1, matching the +9 offset used in the frame script.
    log_ssfr = log_sfr - log_mstar + 9.0
    log_ssfr = np.where(quenched & np.isfinite(log_mstar), SSFR_FLOOR, log_ssfr)
    log_ssfr = np.clip(log_ssfr, SSFR_FLOOR, None)

    orphan = present & (no_stars | ((nbound >= 0) & (nbound < NBOUND_MIN)))

    return dict(log_mstar=log_mstar, log_sfr=log_sfr, log_ssfr=log_ssfr,
                quenched=quenched, orphan=orphan)
