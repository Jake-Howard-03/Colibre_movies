"""
Select galaxies to track in a movie and write their histories in the format
make_movie_frame.py reads.

Everything comes from the master file, track_tools.MASTER_FILE (all galaxies
with log M* > 8 at z=0, every snapshot). Selection and extraction take minutes,
and cuts can use any stored field at any snapshot. Writes track_IDs.txt and
track_histories.npz.

With default arguments the selection reproduces select_and_plot_subsample()
in no_envr.ipynb exactly: 100 galaxies, 5 per bin in 20 equal bins of log M*
between 8 and 12 at z=0, seed 42.

Output order matters: make_movie_frame.py's --n-tracks / --track-stride pick
tracks by position. Even-mass selections are written in ascending bin order and
random ones sorted by mass, so a stride samples across the mass range.

Fields usable in --cut NAME:SNAP:MIN:MAX (empty MIN or MAX = open-ended; SNAP is
a snapshot number, 127 = z=0; MIN/MAX inclusive):
  log_mstar log_sfr log_ssfr log_gas_mass log_m200 r_half_kpc Z_star Z_gas
  log_Z_star log_Z_gas (log10 Z/Zsun, Zsun=0.0134; -inf if Z=0)
  mag_g mag_r g_r is_central n_bound tweb (tweb is z=0 only, SNAP ignored)
The same names work for --spread-by.
log_ssfr is -inf for SFR=0, so a MAX keeps quenched galaxies and a MIN drops them.
A galaxy that doesn't exist at SNAP fails every cut at that SNAP.

Examples:
  python select_tracks_for_movie.py --out d/track_IDs.txt
  python select_tracks_for_movie.py --out d/track_IDs.txt --centrals central --tweb 0 --n 40 --n-bins 8
  python select_tracks_for_movie.py --out d/track_IDs.txt --cut log_ssfr:56:-1: --ssfr-max -2
  python select_tracks_for_movie.py --out d/track_IDs.txt --mass-min 10.9 --mass-max 11.1 --n 25 --spread-by log_Z_star
  python select_tracks_for_movie.py --out d/track_IDs.txt --ids my_ids.txt   # just extract
"""

import argparse
import json
import os

import track_tools  # noqa: F401  (puts simulation_envr/code on sys.path)
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from SWIFT_tools import to_log_mass, to_log_SFR, to_magnitude
from track_tools import APERTURE, H, Z_SUN, MASTER_FILE, SSFR_FLOOR, finalise_histories

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', required=True, help='output TrackId text file')
parser.add_argument('--tracks-out', default=None,
                    help='histories npz; default track_histories.npz next to --out')
parser.add_argument('--master', default=MASTER_FILE)
parser.add_argument('--ids', default=None,
                    help='skip selection: extract exactly these TrackIds')
parser.add_argument('--select-snapshot', type=int, default=127,
                    help='snapshot at which the named cuts and mass bins apply')
parser.add_argument('--mass-min', type=float, default=8.0, help='log10 M*/Msun')
parser.add_argument('--mass-max', type=float, default=12.0)
parser.add_argument('--n', type=int, default=100, help='target number of galaxies')
parser.add_argument('--n-bins', type=int, default=20,
                    help='equal-width log M* bins, n//n_bins per bin; 0 = plain random')
parser.add_argument('--centrals', choices=['all', 'central', 'satellite'], default='all')
parser.add_argument('--ssfr-min', type=float, default=None, help='log10 sSFR/Gyr^-1')
parser.add_argument('--ssfr-max', type=float, default=None, help='log10 sSFR/Gyr^-1')
parser.add_argument('--m200-min', type=float, default=None, help='log10 host M200c/Msun')
parser.add_argument('--m200-max', type=float, default=None)
parser.add_argument('--tweb', default=None,
                    help='z=0 T-web classes to keep, e.g. "0" or "2,3" '
                         '(0 void, 1 sheet, 2 filament, 3 knot)')
parser.add_argument('--cut', action='append', default=[],
                    help='NAME:SNAP:MIN:MAX, repeatable (see above)')
parser.add_argument('--spread-by', default=None, metavar='FIELD',
                    help='instead of mass bins: pick --n galaxies nearest to targets '
                         'evenly spaced in FIELD (at --select-snapshot) between the '
                         '--spread-pct percentiles of the galaxies passing the cuts. '
                         'Output is sorted by FIELD.')
parser.add_argument('--spread-pct', type=float, nargs=2, default=[1.0, 99.0],
                    metavar=('LO', 'HI'))
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

if not os.path.exists(args.master):
    raise SystemExit(f"No master file at {args.master}; build it with code/submit_master_tracks.sh")


# ----------------------------------------------------------------------
# Raw field access: raw(name, snap) -> 1D array over candidate galaxies
# ----------------------------------------------------------------------
class MasterSource:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.snaps = self.f['snap_indices'][:]
        self.track_id = self.f['track_id'][:]
        if self.f.attrs['logm_min_z0'] > args.mass_min and args.select_snapshot == 127:
            raise SystemExit(f"Master only holds log M* > {self.f.attrs['logm_min_z0']} "
                             f"at z=0, so --mass-min can't go lower at snapshot 127")
        self._cache = {}

    def raw(self, name, snap):
        key = (name, snap)
        if key not in self._cache:
            if name == 'tweb_z0':
                v = self.f['tweb_z0'][:]
            elif snap == 127 and 'z0' in self.f:
                v = self.f['z0'][name][:]
            else:
                rows = np.flatnonzero(self.snaps == snap)
                if rows.size == 0:
                    raise SystemExit(f"snapshot {snap} not in master file")
                print(f"  reading {name} at snapshot {snap} (decompresses whole field)")
                v = self.f[name][int(rows[0]), :]
            self._cache[key] = v
        return self._cache[key]

    def redshift(self, snap):
        return float(self.f['redshift'][int(np.flatnonzero(self.snaps == snap)[0])])


def derived(src, name, snap):
    r = lambda k: np.asarray(src.raw(k, snap), dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        if name == 'log_mstar':
            return to_log_mass(r('stellar_mass'))
        if name == 'log_sfr':
            return to_log_SFR(r('star_formation_rate'))
        if name == 'log_ssfr':
            return derived(src, 'log_sfr', snap) - derived(src, 'log_mstar', snap) + 9.0
        if name == 'log_gas_mass':
            return to_log_mass(r('gas_mass'))
        if name == 'log_m200':
            return to_log_mass(r('host_m200c'))
        if name == 'r_half_kpc':
            return r('half_mass_radius_stars') * 1000.0
        if name == 'Z_star':
            return r('stellar_metallicity')
        if name == 'Z_gas':
            return r('gas_metallicity')
        if name == 'log_Z_star':
            return np.log10(r('stellar_metallicity') / Z_SUN)
        if name == 'log_Z_gas':
            return np.log10(r('gas_metallicity') / Z_SUN)
        if name == 'mag_g':
            return to_magnitude(r('lum_g')) - 5 * np.log10(H)
        if name == 'mag_r':
            return to_magnitude(r('lum_r')) - 5 * np.log10(H)
        if name == 'g_r':
            return derived(src, 'mag_g', snap) - derived(src, 'mag_r', snap)
        if name == 'is_central':
            return (np.asarray(src.raw('is_central', snap)) == 1).astype(float)
        if name == 'n_bound':
            return r('n_bound')
        if name == 'tweb':
            return np.asarray(src.raw('tweb_z0', 127), dtype=float)
    raise SystemExit(f"Unknown field '{name}' (see --help for the list)")


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
S = args.select_snapshot
src = MasterSource(args.master)
track_id = src.track_id
print(f"Candidates: {track_id.size}, selection snapshot {S} (z={src.redshift(S):.3f})")

log_mstar = derived(src, 'log_mstar', S)

if args.ids:
    want = np.atleast_1d(np.loadtxt(args.ids, dtype=np.int64))
    lookup = {t: i for i, t in enumerate(track_id)}
    absent = [t for t in want if t not in lookup]
    if absent:
        raise SystemExit(f"{len(absent)} TrackIds not in master (e.g. {absent[:5]}); "
                         f"they weren't log M* > 8 at z=0.")
    selected = np.array([lookup[t] for t in want])
    mask = np.zeros(track_id.size, bool)
    mask[selected] = True
else:
    cuts = [('log_mstar', S, args.mass_min, None)]  # strict > handled below
    if args.centrals != 'all':
        cuts.append(('is_central', S, *((0.5, None) if args.centrals == 'central' else (None, 0.5))))
    if args.ssfr_min is not None or args.ssfr_max is not None:
        cuts.append(('log_ssfr', S, args.ssfr_min, args.ssfr_max))
    if args.m200_min is not None or args.m200_max is not None:
        cuts.append(('log_m200', S, args.m200_min, args.m200_max))
    for c in args.cut:
        name, snap, lo, hi = c.split(':')
        cuts.append((name, int(snap), float(lo) if lo else None, float(hi) if hi else None))

    mask = (log_mstar > args.mass_min) & (log_mstar < args.mass_max)
    for name, snap, lo, hi in cuts[1:]:
        v = derived(src, name, snap)
        if lo is not None:
            mask &= v >= lo
        if hi is not None:
            mask &= v <= hi
        print(f"  after {name}@{snap} in [{lo}, {hi}]: {mask.sum()}")
    if args.tweb is not None:
        keep = [int(k) for k in args.tweb.split(',')]
        mask &= np.isin(derived(src, 'tweb', 127), keep)
        print(f"  after tweb in {keep}: {mask.sum()}")
    print(f"{mask.sum()} galaxies pass the cuts")
    if mask.sum() == 0:
        raise SystemExit("Nothing passes the cuts")

    masked_indices = np.where(mask)[0]
    rng = np.random.default_rng(args.seed)
    if args.spread_by:
        v = derived(src, args.spread_by, S)
        cand = masked_indices[np.isfinite(v[masked_indices])]
        if cand.size < args.n:
            raise SystemExit(f"Only {cand.size} candidates with finite {args.spread_by}")
        vals = v[cand]
        lo, hi = np.percentile(vals, args.spread_pct)
        targets = np.linspace(lo, hi, args.n)
        print(f"Spreading {args.n} galaxies in {args.spread_by} from {lo:.3f} to {hi:.3f} "
              f"(percentiles {args.spread_pct[0]:g}-{args.spread_pct[1]:g} of {cand.size})")
        taken = np.zeros(cand.size, bool)
        chosen = []
        for t in targets:
            d = np.abs(vals - t)
            d[taken] = np.inf
            k = int(np.argmin(d))
            taken[k] = True
            chosen.append(k)
            print(f"  target {t:8.3f}  picked {vals[k]:8.3f}  (TrackId {track_id[cand[k]]})")
        chosen = np.array(chosen)
        selected = cand[chosen[np.argsort(vals[chosen], kind='stable')]]
    elif args.n_bins > 0:
        bin_edges = np.linspace(args.mass_min, args.mass_max, args.n_bins + 1)
        bin_idx = np.digitize(log_mstar[mask], bin_edges) - 1
        per_bin = max(1, args.n // args.n_bins)
        picked = []
        for b in range(args.n_bins):
            in_bin = np.where(bin_idx == b)[0]
            if len(in_bin) == 0:
                print(f"  bin {b}: empty")
                continue
            n_pick = min(per_bin, len(in_bin))
            if n_pick < per_bin:
                print(f"  bin {b}: only {len(in_bin)} available")
            picked.extend(rng.choice(in_bin, size=n_pick, replace=False))
        picked = np.array(picked, dtype=int)
    else:
        n_pick = min(args.n, mask.sum())
        picked = rng.choice(mask.sum(), size=n_pick, replace=False)
        picked = picked[np.argsort(log_mstar[mask][picked])]
    if not args.spread_by:
        selected = masked_indices[picked]

selected_ids = track_id[selected]
if np.unique(selected_ids).size != selected_ids.size:
    raise ValueError("Duplicate TrackIds in selection")
is_cen_sel = derived(src, 'is_central', S)[selected] == 1
print(f"Selected {len(selected)} galaxies: {is_cen_sel.sum()} central, "
      f"{(~is_cen_sel).sum()} satellite (status at snapshot {S})")

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savetxt(args.out, selected_ids, fmt='%d')
stem = os.path.splitext(args.out)[0]
with open(stem + '_selection.json', 'w') as f:
    json.dump({**vars(args), 'redshift': src.redshift(S),
               'n_selected': int(len(selected)), 'n_passing_cuts': int(mask.sum())},
              f, indent=2)

# Quick-look plot at the selection snapshot: mass vs the --spread-by field if
# given, otherwise the mass-sSFR plane.
base = log_mstar > 8
if args.spread_by:
    yq = derived(src, args.spread_by, S)
    ok = base & np.isfinite(yq)
    y_range = list(np.percentile(yq[ok], [0.2, 99.8]))
    y_label = args.spread_by
else:
    yq = np.nan_to_num(np.clip(derived(src, 'log_ssfr', S), SSFR_FLOOR, None),
                       nan=SSFR_FLOOR)
    ok = base
    y_range = [-5, 2]
    y_label = r'$\log_{10}(\mathrm{sSFR}/\mathrm{Gyr}^{-1})$ (SFR=0 at floor)'
fig, ax = plt.subplots(figsize=(8, 6.4), dpi=150)
ax.hist2d(log_mstar[ok], yq[ok], bins=250, range=[[8, 12], y_range],
          cmap='YlOrRd', norm=LogNorm())
ax.scatter(log_mstar[selected][is_cen_sel], yq[selected][is_cen_sel], s=20,
           c='blue', edgecolors='black', linewidths=0.4, label='central', zorder=5)
ax.scatter(log_mstar[selected][~is_cen_sel], yq[selected][~is_cen_sel], s=20,
           facecolors='none', edgecolors='blue', linewidths=1.0, label='satellite', zorder=5)
ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(y_label)
ax.set_title(f'{len(selected)} selected, snapshot {S}, z={src.redshift(S):.2f}')
ax.legend(loc='lower left')
fig.savefig(stem + '_selection.png')
print(f"Wrote {args.out}, {stem}_selection.json, {stem}_selection.png")

# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
order = np.argsort(selected)              # h5py wants increasing indices
cols = selected[order]
inv = np.empty_like(order)
inv[order] = np.arange(order.size)
f = src.f

def col(name):
    return f[name][:, cols][:, inv]

raw = {k: col(k) for k in ['stellar_mass', 'star_formation_rate', 'n_bound',
                           'is_central', 'present', 'gas_mass', 'host_m200c',
                           'half_mass_radius_stars', 'stellar_metallicity',
                           'gas_metallicity', 'lum_g', 'lum_r']}
present = raw['present'].astype(bool)
mstar = np.where(present, raw['stellar_mass'], np.nan)
sfr = np.where(present, raw['star_formation_rate'], np.nan)
fin = finalise_histories(mstar, sfr, raw['n_bound'], present)

with np.errstate(divide='ignore', invalid='ignore'):
    extra = dict(
        log_gas_mass=to_log_mass(raw['gas_mass'].astype(np.float64)),
        log_host_m200c=to_log_mass(raw['host_m200c'].astype(np.float64)),
        r_half_kpc=raw['half_mass_radius_stars'] * 1000.0,
        stellar_metallicity=raw['stellar_metallicity'],
        gas_metallicity=raw['gas_metallicity'],
        mag_g=to_magnitude(raw['lum_g'].astype(np.float64)) - 5 * np.log10(H),
        mag_r=to_magnitude(raw['lum_r'].astype(np.float64)) - 5 * np.log10(H),
    )

tracks_out = args.tracks_out or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                             'track_histories.npz')
np.savez(
    tracks_out,
    track_ids=selected_ids,
    snap_indices=src.snaps,
    redshift=f['redshift'][:],
    log_mstar=fin['log_mstar'], log_sfr=fin['log_sfr'], log_ssfr=fin['log_ssfr'],
    present=present, quenched=fin['quenched'], orphan=fin['orphan'],
    nbound=raw['n_bound'], is_central=raw['is_central'] == 1,
    ssfr_floor=SSFR_FLOOR, aperture=APERTURE,
    tweb_z0=f['tweb_z0'][:][selected],
    **extra,
)
print(f"Wrote {tracks_out}: {present.shape[0]} snapshots x {present.shape[1]} tracks")
