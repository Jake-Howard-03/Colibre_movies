"""
Stack the per-snapshot parts from build_master_tracks.py into one HDF5 file.

Layout of the output (n_tracks = galaxies with log M* > 8 at z=0):

  /track_id            (n_tracks,)        z=0 TrackIds, SOAP z=0 row order
  /z0_row              (n_tracks,)        their row in the z=0 SOAP catalogue
  /tweb_z0             (n_tracks,) int8   0 void, 1 sheet, 2 filament, 3 knot
  /centre_of_mass_z0   (n_tracks, 3)
  /snap_indices, /redshift  (n_snaps,)
  /<field>             (n_snaps, n_tracks) raw SOAP values, NaN / -1 if absent
                       attrs: units, comoving
  /z0/<field>          (n_tracks,)        copy of the last snapshot row, so
                       selections at z=0 don't have to decompress everything

Fields: see track_tools.MASTER_SNAP_FIELDS.
"""

import argparse
import os
import shutil

import h5py
import numpy as np

from track_tools import (MASTER_FILE, MASTER_IDS, MASTER_PARTS, MASTER_SNAP_FIELDS,
                         MASTER_FLOAT_FIELDS, APERTURE, SIM)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--snap-first', type=int, default=2)
parser.add_argument('--snap-last', type=int, default=127)
parser.add_argument('--ids-file', default=MASTER_IDS)
parser.add_argument('--parts-dir', default=MASTER_PARTS)
parser.add_argument('--output', default=MASTER_FILE)
parser.add_argument('--delete-parts', action='store_true',
                    help='remove the parts directory after a successful merge')
args = parser.parse_args()

snaps = np.arange(args.snap_first, args.snap_last + 1)
paths = [os.path.join(args.parts_dir, f"snap_{s:04d}.npz") for s in snaps]
missing = [s for s, p in zip(snaps, paths) if not os.path.exists(p)]
if missing:
    raise SystemExit(f"Missing parts for snapshots: {missing}\n"
                     f"Resubmit those with build_master_tracks.py --snap N")

ids = np.load(args.ids_file)
n = ids['track_id'].size
n_snaps = snaps.size
print(f"Merging {n_snaps} snapshots x {n} tracks")

redshift = np.empty(n_snaps)
units = {}
for i, p in enumerate(paths):
    with np.load(p) as part:
        if part['present'].size != n:
            raise ValueError(f"{p} has {part['present'].size} tracks, expected {n}")
        redshift[i] = part['redshift']
        if i == n_snaps - 1:
            for line in part['units']:
                k, u, c = str(line).split('|')
                units[k] = (u, c)

# which SOAP field each stored field's units came from
field_src = {name: field for name, (_, field, _) in MASTER_FLOAT_FIELDS.items()}
field_src['host_m200c'] = 'host_m200c'

tmp = args.output + '.tmp'
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with h5py.File(tmp, 'w') as f:
    f['track_id'] = ids['track_id']
    f['z0_row'] = ids['z0_row']
    f['tweb_z0'] = ids['tweb_z0']
    f['centre_of_mass_z0'] = ids['centre_of_mass_z0']
    f['centre_of_mass_z0'].attrs['units'] = str(ids['com_units'][0])
    f['snap_indices'] = snaps
    f['redshift'] = redshift
    f.attrs['logm_min_z0'] = float(ids['logm_min'])
    f.attrs['aperture'] = APERTURE
    f.attrs['simulation'] = '/'.join(SIM.values())
    z0 = f.create_group('z0')

    chunk = (n_snaps, min(4096, n))
    for field in MASTER_SNAP_FIELDS:
        stack = None
        for i, p in enumerate(paths):
            with np.load(p) as part:
                row = part[field]
            if stack is None:
                stack = np.empty((n_snaps, n), dtype=row.dtype)
            stack[i] = row
        ds = f.create_dataset(field, data=stack, chunks=chunk,
                              compression='gzip', compression_opts=4, shuffle=True)
        if field in field_src and field_src[field] in units:
            ds.attrs['units'], ds.attrs['comoving'] = units[field_src[field]]
        if snaps[-1] == 127:
            z0[field] = stack[-1]
        if stack.dtype.kind == 'f':
            frac = np.isfinite(stack).mean()
        elif stack.dtype.kind == 'b':
            frac = stack.mean()
        else:
            frac = (stack != -1).mean()
        print(f"  {field:24s} {str(stack.dtype):8s} filled {frac:.3f}")
        del stack

os.replace(tmp, args.output)
print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1e9:.2f} GB)")

if args.delete_parts:
    shutil.rmtree(args.parts_dir)
    print(f"Removed {args.parts_dir}")
