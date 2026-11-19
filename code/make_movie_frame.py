"""
Make a single movie frame from a SOAP catalogue, optionally overlaying the
evolutionary tracks of a set of galaxies selected at z=0.

Normally run by submit_movie.sh / submit_movie_set.sh. By hand:
    python make_movie_frame.py 42 --out-dir DIR
    python make_movie_frame.py 42 --out-dir DIR --tracks --tracks-file T.npz --tail 20
"""

import argparse
import os
import sys

import track_tools  # noqa: F401  (puts simulation_envr/code on sys.path)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from SWIFT_tools import (
    load_SOAP_snapshot,
    load_array_field,
    load_luminosity,
    to_log_mass,
    to_magnitude,
    to_log_SFR,
)
from plotting_funcs import style_ticks
from track_tools import APERTURE, SIM, H, Z_SUN

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Only show track points that would also survive the background mass cut,
# so the trails never wander into a region the histogram has emptied.
MASS_MIN = 8.0

# Trail appearance
TRAIL_COLOUR = '#08306b'      # dark blue, reads against YlOrRd
HEAD_COLOUR = '#08306b'
TRAIL_LW = 1.4
HEAD_SIZE = 45
ALPHA_MAX = 0.9
ALPHA_MIN = 0.05

# Quenched galaxies are all pinned to the same sSFR floor by the track
# builder, so spread them slightly to stop them stacking into a fake line.
FLOOR_JITTER = 0.08
JITTER_SEED = 42

# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('index', type=int, help='snapshot index for this frame')
parser.add_argument('--plot-syntax', default='mass-sSFR',
                    choices=['col-mag', 'mass-sSFR', 'mass-SFR', 'mass-Zstar',
                             'mass-colour'])
parser.add_argument('--tracks', action='store_true',
                    help='overlay z=0 galaxy tracks')
parser.add_argument('--tracks-file', action='append', default=None,
                    help='track histories npz; repeat (paired with --out-dir) '
                         'to write several frames from one snapshot load. '
                         "'none' gives that frame no overlay. "
                         'Required with --tracks')
parser.add_argument('--tail', type=int, default=15,
                    help='trail length in snapshots')
parser.add_argument('--n-tracks', type=int, default=None,
                    help='use only the first N tracks in the file')
parser.add_argument('--track-stride', type=int, default=1,
                    help='use every Nth track (applied after --n-tracks)')
parser.add_argument('--min-run', type=int, default=2,
                    help='require this many consecutive valid snapshots '
                         'before a track is drawn, to stop flickering')
parser.add_argument('--out-dir', action='append', required=True,
                    help='output directory; repeat once per --tracks-file')
parser.add_argument('--norm', choices=['fixed', 'auto'], default='fixed',
                    help="'fixed': colour scale capped at the hard-coded z=0 "
                         "vmax, so density visibly builds up; 'auto': each "
                         "frame scaled to its own maximum")
args = parser.parse_args()

index = args.index
snapshot = f"{index:04d}"
plot_syntax = args.plot_syntax
print(f"Processing snapshot {snapshot}")

if args.tracks and plot_syntax == 'col-mag':
    sys.exit("Track overlay is not defined for the col-mag plot")

out_dirs = args.out_dir
if args.tracks:
    if not args.tracks_file:
        sys.exit("--tracks needs --tracks-file")
    tracks_files = [None if t.lower() == 'none' else t for t in args.tracks_file]
    if len(tracks_files) != len(out_dirs):
        sys.exit(f"Got {len(tracks_files)} --tracks-file but {len(out_dirs)} "
                 f"--out-dir; give one out dir per tracks file")
else:
    if len(out_dirs) != 1:
        sys.exit("Several --out-dir only make sense with --tracks")
    tracks_files = [None]

# ----------------------------------------------------------------------
# Load snapshot
# ----------------------------------------------------------------------
data, metadata = load_SOAP_snapshot(snapshot=snapshot, **SIM)

needs_colour = plot_syntax in ('col-mag', 'mass-colour')
needs_sfr = plot_syntax in ('mass-sSFR', 'mass-SFR')
needs_zstar = plot_syntax == 'mass-Zstar'

stellar_mass = load_array_field(data, APERTURE, 'stellar_mass').value
log_stellar_mass = to_log_mass(stellar_mass)

if needs_colour:
    lum_r = load_luminosity(data, APERTURE, 'r').value
    lum_g = load_luminosity(data, APERTURE, 'g').value
    mag_r = to_magnitude(lum_r) - 5 * np.log10(H)
    mag_g = to_magnitude(lum_g) - 5 * np.log10(H)

if needs_sfr:
    sfr = load_array_field(data, APERTURE, 'star_formation_rate').value
    with np.errstate(divide='ignore'):
        log_sfr = to_log_SFR(sfr)

if needs_zstar:
    z_star = load_array_field(data, APERTURE,
                              'stellar_mass_fraction_in_metals').value

# ----------------------------------------------------------------------
# Build the background
# ----------------------------------------------------------------------
if plot_syntax == 'col-mag':
    mask = (log_stellar_mass > MASS_MIN) & (mag_r < -17)
    x = mag_g[mask] - mag_r[mask]
    y = mag_r[mask]
    plot_range = [[-0.6, 1.3], [-24.5, -10]]
    vmax = 536
    xlabel = r'$\mathrm{M}_g - \mathrm{M}_r$'
    ylabel = r'$\mathrm{M}_r - 5\log h$'

elif plot_syntax == 'mass-sSFR':
    mask = log_stellar_mass > MASS_MIN
    x = log_stellar_mass[mask]
    y = log_sfr[mask] - log_stellar_mass[mask] + 9
    plot_range = [[8, 12], [-5, 2]]
    vmax = 425
    xlabel = r'$\log_{10}(M_\star/M_\odot)$'
    ylabel = r'$\log_{10}(\mathrm{sSFR}/\mathrm{Gyr}^{-1})$'

elif plot_syntax == 'mass-SFR':
    mask = log_stellar_mass > MASS_MIN
    x = log_stellar_mass[mask]
    y = log_sfr[mask]
    plot_range = [[8, 12], [-4, 2.5]]
    vmax = 694
    xlabel = r'$\log_{10}(M_\star/M_\odot)$'
    ylabel = r'$\log_{10}(\mathrm{SFR}/(M_\odot\,\mathrm{yr}^{-1}))$'

elif plot_syntax == 'mass-Zstar':
    mask = (log_stellar_mass > MASS_MIN) & (z_star > 0)
    x = log_stellar_mass[mask]
    y = np.log10(z_star[mask] / Z_SUN)
    # -3 keeps >99.9% of log M*>8 galaxies on the plot even at z~7.
    plot_range = [[8, 12], [-3.0, 0.5]]
    vmax = 1566   # z=0 max of this hist2d (from the master file's /z0)
    xlabel = r'$\log_{10}(M_\star/M_\odot)$'
    ylabel = r'$\log_{10}(Z_\star/Z_\odot)$'

elif plot_syntax == 'mass-colour':
    with np.errstate(invalid='ignore'):
        g_r = mag_g - mag_r
    mask = (log_stellar_mass > MASS_MIN) & np.isfinite(g_r)
    x = log_stellar_mass[mask]
    y = g_r[mask]
    # Rest-frame, dust-free. -0.5 keeps >99.9% of log M*>8 galaxies on the
    # plot even at z~7 (at z=0 the 0.1-99.9% range is -0.29..0.80).
    plot_range = [[8, 12], [-0.5, 0.9]]
    vmax = 4848   # z=0 max of this hist2d (from the master file's /z0)
    xlabel = r'$\log_{10}(M_\star/M_\odot)$'
    ylabel = r'$\mathrm{M}_g - \mathrm{M}_r$'

# Bottom-left is where the dense low-mass galaxies sit on these plots.
legend_loc = 'lower right' if plot_syntax in ('mass-Zstar', 'mass-colour') else 'lower left'

print(f'Number of galaxies: {mask.sum()}')

# ----------------------------------------------------------------------
# Track overlay
# ----------------------------------------------------------------------
def draw_tracks(ax, tracks_file):
    tr = np.load(tracks_file)

    snap_indices = tr['snap_indices']
    rows = np.flatnonzero(snap_indices == index)
    if rows.size == 0:
        raise ValueError(
            f"snapshot {index} is not in {tracks_file} "
            f"(covers {snap_indices.min()}..{snap_indices.max()})"
        )
    row = int(rows[0])

    # Guard against the tracks and the frames drifting out of step.
    dz = abs(tr['redshift'][row] - metadata.redshift)
    if dz > 1e-3:
        raise ValueError(
            f"redshift mismatch at snapshot {snapshot}: "
            f"npz has {tr['redshift'][row]:.4f}, snapshot has "
            f"{metadata.redshift:.4f}"
        )

    log_mstar_t = tr['log_mstar']
    if plot_syntax == 'mass-sSFR':
        y_track = tr['log_ssfr']
    elif plot_syntax == 'mass-SFR':
        y_track = tr['log_sfr']
    elif plot_syntax == 'mass-colour':
        if 'mag_g' not in tr.files:
            raise ValueError(
                f"{tracks_file} has no mag_g/mag_r; it must come from the "
                f"master-file route (select_tracks_for_movie.py --source master)")
        with np.errstate(invalid='ignore'):
            y_track = (tr['mag_g'] - tr['mag_r']).astype(np.float64)
    else:  # mass-Zstar
        if 'stellar_metallicity' not in tr.files:
            raise ValueError(
                f"{tracks_file} has no stellar_metallicity; it must come "
                f"from the master-file route (select_tracks_for_movie.py --source master)")
        with np.errstate(divide='ignore', invalid='ignore'):
            y_track = np.log10(tr['stellar_metallicity'].astype(np.float64) / Z_SUN)
    present = tr['present']
    quenched = tr['quenched']
    orphan = tr['orphan']
    is_central = tr['is_central']
    n_tracks_all = log_mstar_t.shape[1]

    # Optional subsetting
    sel = np.arange(n_tracks_all)
    if args.n_tracks is not None:
        sel = sel[:args.n_tracks]
    if args.track_stride > 1:
        sel = sel[::args.track_stride]
    print(f"Overlaying {len(sel)} of {n_tracks_all} tracks")

    log_mstar_t = log_mstar_t[:, sel]
    y_track = y_track[:, sel].copy()
    present = present[:, sel]
    quenched = quenched[:, sel]
    orphan = orphan[:, sel]
    is_central = is_central[:, sel]

    # A galaxy is drawn only where it is matched, resolved, above the same
    # mass cut as the background, and has a finite position.
    valid = (
        present
        & ~orphan
        & (log_mstar_t > MASS_MIN)
        & np.isfinite(log_mstar_t)
        & np.isfinite(y_track)
    )

    # Suppress single-snapshot appearances so tracks don't strobe.
    if args.min_run > 1:
        run = np.ones_like(valid)
        for k in range(1, args.min_run):
            run[k:] &= valid[:-k]
        run[:args.min_run - 1] = False
        valid &= run

    # Spread the quenched pile-up at the sSFR floor.
    if plot_syntax == 'mass-sSFR':
        rng = np.random.default_rng(JITTER_SEED)
        offsets = rng.uniform(-FLOOR_JITTER, FLOOR_JITTER, size=len(sel))
        y_track = np.where(quenched, y_track + offsets[None, :], y_track)

    xs = np.where(valid, log_mstar_t, np.nan)
    ys = np.where(valid, y_track, np.nan)

    # --- trails ---
    start = max(0, row - args.tail)
    seg_list = []
    seg_alpha = []
    n_steps = row - start
    for k in range(n_steps):
        j = start + k
        p0x, p0y = xs[j], ys[j]
        p1x, p1y = xs[j + 1], ys[j + 1]
        good = np.isfinite(p0x) & np.isfinite(p0y) & np.isfinite(p1x) & np.isfinite(p1y)
        if not good.any():
            continue
        # Alpha ramps from faint at the oldest step to strong at the head.
        frac = (k + 1) / n_steps
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * frac
        segs = np.stack(
            [np.column_stack([p0x[good], p0y[good]]),
             np.column_stack([p1x[good], p1y[good]])],
            axis=1,
        )
        seg_list.append(segs)
        seg_alpha.append(np.full(segs.shape[0], alpha))

    if seg_list:
        segs = np.concatenate(seg_list, axis=0)
        alphas = np.concatenate(seg_alpha)
        rgba = np.tile(
            np.array(plt.matplotlib.colors.to_rgba(TRAIL_COLOUR)),
            (len(alphas), 1),
        )
        rgba[:, 3] = alphas
        ax.add_collection(
            LineCollection(segs, colors=rgba, linewidths=TRAIL_LW,
                           capstyle='round', zorder=3)
        )

    # --- heads: filled = central, open = satellite ---
    head = valid[row]
    cen = head & is_central[row]
    sat = head & ~is_central[row]

    ax.scatter(xs[row][cen], ys[row][cen],
               s=HEAD_SIZE, facecolors=HEAD_COLOUR, edgecolors='white',
               linewidths=0.8, zorder=5, label='Central')
    ax.scatter(xs[row][sat], ys[row][sat],
               s=HEAD_SIZE, facecolors='none', edgecolors=HEAD_COLOUR,
               linewidths=1.4, zorder=5, label='Satellite')

    ax.legend(loc=legend_loc, fontsize=13, framealpha=0.8,
              markerscale=1.2).set_zorder(6)

    print(f"heads drawn: {cen.sum()} central, {sat.sum()} satellite")
    tr.close()


# ----------------------------------------------------------------------
# One frame per (tracks file, out dir) pair; the snapshot is loaded once
# ----------------------------------------------------------------------
for tracks_file, out_dir in zip(tracks_files, out_dirs):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist2d(
        x, y,
        bins=250,
        range=plot_range,
        cmap='YlOrRd',
        norm=LogNorm(vmin=1, vmax=vmax if args.norm == 'fixed' else None),
    )
    if tracks_file is not None:
        draw_tracks(ax, tracks_file)

    style_ticks(ax)
    ax.set_xlim(*plot_range[0])
    ax.set_ylim(*plot_range[1])
    if plot_syntax == 'col-mag':
        ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.text(0.98, 0.98, f"z={metadata.redshift:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=22,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"frame_{snapshot}.png")

    plt.savefig(outfile, dpi=300)
    plt.close("all")

    print(f"Finished: {outfile}")

del data, metadata
