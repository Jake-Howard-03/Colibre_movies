"""
Build the master track histories, one snapshot at a time.

Two modes (run by code/submit_master_tracks.sh):

  python build_master_tracks.py --make-ids
      At z=0 (snapshot 0127), take every subhalo with log M* > LOGM_MIN in the
      50 kpc exclusive sphere. Save their TrackIds (in SOAP row order, which
      keeps random selections reproducible against the old SOAP-based ones),
      centre of mass and z=0 T-web class to track_tools.MASTER_IDS.

  python build_master_tracks.py --snap N
      Find those TrackIds in snapshot N and write one length-n_tracks array
      per field to parts/snap_NNNN.npz. Run as a Slurm array over 2..127,
      then merge_master_tracks.py stacks the parts into one HDF5 file.
"""

import argparse
import os

import track_tools  # noqa: F401  (puts simulation_envr/code on sys.path)
import numpy as np

from SWIFT_tools import load_SOAP_snapshot, load_array_field, broadcast_to_host, to_log_mass
from Tweb_tools import load_Tweb_data, convert_grid_to_Mpc, assign_environments
from track_tools import (
    APERTURE, SIM, MASTER_DIR, MASTER_IDS, MASTER_PARTS, MASTER_FLOAT_FIELDS,
    match_track_ids,
)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--make-ids', action='store_true')
mode.add_argument('--snap', type=int)
parser.add_argument('--logm-min', type=float, default=8.0)
parser.add_argument('--ids-file', default=MASTER_IDS)
parser.add_argument('--parts-dir', default=MASTER_PARTS)
args = parser.parse_args()


def units_of(arr):
    return str(getattr(arr, 'units', '')), str(getattr(arr, 'comoving', ''))


def save_atomic(path, **arrays):
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
if args.make_ids:
    data, meta = load_SOAP_snapshot(snapshot='0127', **SIM)
    track_id = load_array_field(data, 'input_halos_hbtplus', 'track_id').value
    log_mstar = to_log_mass(load_array_field(data, APERTURE, 'stellar_mass').value)
    rows = np.flatnonzero(log_mstar > args.logm_min)
    print(f"{rows.size} subhalos with log M* > {args.logm_min} at z={meta.redshift:.3f}")

    com = load_array_field(data, APERTURE, 'centre_of_mass')
    com_units = units_of(com)
    com = com.value[rows]

    # Same recipe as envr_non_obs.ipynb: T-web class at each galaxy's own position.
    Xg, Yg, Zg, Cweb = load_Tweb_data()
    _, _, _, cell_width, n_cells = convert_grid_to_Mpc(Xg, Yg, Zg)
    tweb = assign_environments(com, cell_width, n_cells, Cweb).astype(np.int8)
    print("T-web counts (void, sheet, filament, knot):",
          [int((tweb == k).sum()) for k in range(4)])

    ids = track_id[rows]
    if np.unique(ids).size != ids.size:
        raise ValueError("Duplicate TrackIds at z=0")

    os.makedirs(os.path.dirname(args.ids_file), exist_ok=True)
    save_atomic(args.ids_file, track_id=ids, z0_row=rows,
                centre_of_mass_z0=com.astype(np.float32), tweb_z0=tweb,
                logm_min=args.logm_min, com_units=np.array(com_units))
    print(f"Saved {args.ids_file}")

# ----------------------------------------------------------------------
else:
    snapshot = f"{args.snap:04d}"
    ids = np.load(args.ids_file)['track_id']
    n = ids.size

    data, meta = load_SOAP_snapshot(snapshot=snapshot, **SIM)
    found, rows = match_track_ids(
        load_array_field(data, 'input_halos_hbtplus', 'track_id').value, ids)
    print(f"snapshot {snapshot}  z={meta.redshift:.3f}  matched {found.sum()}/{n}")

    out = {'redshift': np.float64(meta.redshift), 'snap': np.int32(args.snap),
           'present': found}
    units = {}

    # Cache 2D fields (luminosity) so g and r come from one read.
    cache = {}
    for name, (group, field, col) in MASTER_FLOAT_FIELDS.items():
        if (group, field) not in cache:
            arr = load_array_field(data, group, field)
            units[field] = units_of(arr)
            cache[(group, field)] = arr.value
        vals = cache[(group, field)]
        col_vals = vals[rows] if col is None else vals[rows, col]
        full = np.full(n, np.nan, dtype=np.float32)
        full[found] = col_vals
        out[name] = full
        if col is None:
            del cache[(group, field)]
    cache.clear()

    # Host M200c: centrals keep their own, satellites take their central's,
    # via SOAP/HostHaloIndex (same as the environment notebooks).
    m200 = load_array_field(data, 'spherical_overdensity_200_crit', 'total_mass')
    units['host_m200c'] = units_of(m200)
    hhi = load_array_field(data, 'soap', 'host_halo_index').value
    host_m200 = broadcast_to_host(m200.value, hhi)
    full = np.full(n, np.nan, dtype=np.float32)
    full[found] = host_m200[rows]
    out['host_m200c'] = full
    del m200, hhi, host_m200

    nb = np.full(n, -1, dtype=np.int32)
    nb[found] = load_array_field(data, 'input_halos', 'number_of_bound_particles').value[rows]
    out['n_bound'] = nb

    ic = np.full(n, -1, dtype=np.int8)
    ic[found] = (load_array_field(data, 'input_halos', 'is_central').value[rows] == 1)
    out['is_central'] = ic

    out['units'] = np.array([f"{k}|{u}|{c}" for k, (u, c) in units.items()])

    os.makedirs(args.parts_dir, exist_ok=True)
    path = os.path.join(args.parts_dir, f"snap_{snapshot}.npz")
    save_atomic(path, **out)
    print(f"Saved {path}")
