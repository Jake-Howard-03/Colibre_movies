---
name: make-movie
description: Make a redshift-evolution movie of a COLIBRE scatter/density plot (mass-sSFR, mass-SFR, colour-magnitude), optionally with tracks of selected galaxies, by submitting the Slurm pipeline. Use whenever Jake asks for an animation/movie/frames of a graph growing over redshift, to select galaxies to track, to extract track histories, or to re-stitch/redo a movie.
---

# Making redshift-evolution movies

Jake says *which galaxies* and *which graph*; you translate that into one
`code/submit_movie.sh` call and submit it. He has authorised you to submit these
jobs (Nov 2026), so you don't need to ask first. Do ask if the request is
genuinely ambiguous, for example "the massive ones" with no threshold.

All paths below are relative to `/cosma8/data/dp004/dc-howa2/colibre_movies`.
The scripts import `SWIFT_tools`, `Tweb_tools` and `plotting_funcs` from
`/cosma8/data/dp004/dc-howa2/simulation_envr/code` (via `track_tools.ENVR_CODE`).

## Pipeline

**Master track file.** `data/master_tracks/track_histories_logM8.hdf5`
holds the history, back to snapshot 2, of every galaxy with log M* > 8 at z=0
(~2.4M galaxies). It is built once with `code/submit_master_tracks.sh`; see
"Master file" below. Every movie with tracks reads it:

| step | script | output | resources |
|---|---|---|---|
| 1 extract | `code/select_tracks_for_movie.py` | `data/NAME/track_IDs.txt` (+ `_selection.json`, `_selection.png`) and `track_histories.npz` | 16G, minutes |
| 2 frames | `code/make_movie_frame.py` (array, one task per snapshot) | `frames/NAME/frame_XXXX.png` | 16G, ~5–10 min each |
| 3 stitch | ffmpeg 7.0.1, libx264, yuv420p | `animations/NAME.mp4` | 4G, minutes |

Only galaxies in the master file can be tracked (log M* > 8 at z=0). The old
per-movie SOAP route was removed in Nov 2026; to track anything else, rebuild
the master file with a lower `--logm-min` (see "Master file").

The steps are chained with `afterok`, using `--kill-on-invalid-dep=yes`. If
anything fails, every later step is cancelled. The submit command and its
settings are recorded in `data/NAME/movie_config.txt`. (Configs from before the
Nov 2026 move still show old `simulation_envr/...` paths.)

Everything uses COLIBRE `L400_m7` / `Thermal`. Snapshots run 2–127 (127 = z=0).
Quantities come from `exclusive_sphere_50kpc`, and the background histogram
only includes galaxies with log M* > 8. Shared constants and the
log/floor/orphan post-processing are in `code/track_tools.py`.

## Turning a request into a command

```bash
cd /cosma8/data/dp004/dc-howa2/colibre_movies
code/submit_movie.sh NAME [--plot TYPE] [--select "..."] [--frame-args "..."] [other options]
```

1. **Pick NAME.** Use a short snake_case name that describes the content, e.g.
   `mass_sSFR_centrals_even50` or `mass_sSFR_voids_quenched`. The script
   refuses a NAME that already exists, so choose a new one. Use `--overwrite`
   only if Jake asks to redo that movie.
2. **Graph (`--plot`):**
   - `mass-sSFR` (default): x = log M*, y = log sSFR [Gyr⁻¹]
   - `mass-SFR`
   - `mass-Zstar`: x = log M*, y = log10(Z*/Z☉) with Z☉ = 0.0134, axis −3 to
     0.5. Tracks read `stellar_metallicity` from the npz.
   - `mass-colour`: x = log M*, y = rest-frame dust-free g−r, axis −0.5 to 0.9
     (vmax 4848). Tracks read `mag_g`/`mag_r` from the npz.
   - `col-mag`: g−r vs M_r − 5 log h. **Background only, so it needs `--no-tracks`.**
     The master histories do store `mag_g`/`mag_r`, so adding the overlay is
     possible; see Limitations.
3. **Galaxies (`--select "..."`).** These are flags for `select_tracks_for_movie.py`.
   The named cuts and the mass bins apply at `--select-snapshot` (default 127,
   z=0).
   - `--n 100 --n-bins 20`: N galaxies, spread evenly over equal log M* bins
     (n//n_bins per bin). `--n-bins 0` gives a plain random draw instead.
   - `--mass-min 8 --mass-max 12`. `--mass-min` can't go below 8 at z=0.
   - `--centrals all|central|satellite`
   - `--ssfr-min X` / `--ssfr-max X` (log Gyr⁻¹). `--ssfr-max` keeps SFR=0
     galaxies, so for "quenched" use `--ssfr-max -2`; `--ssfr-min` drops them.
   - `--m200-min X` / `--m200-max X`: log host M200c in Msun. Satellites take
     their central's value.
   - `--tweb 0,1,2,3`: z=0 T-web class at the galaxy's own position (0 void,
     1 sheet, 2 filament, 3 knot).
   - `--select-snapshot N`: apply the named cuts and the mass binning at
     snapshot N instead of z=0.
   - `--cut NAME:SNAP:MIN:MAX` (repeatable; leave MIN or MAX empty for
     open-ended; bounds are inclusive). This works for any field at any
     snapshot:
     - fields: `log_mstar log_sfr log_ssfr log_gas_mass log_m200 r_half_kpc Z_star Z_gas log_Z_star log_Z_gas mag_g mag_r g_r is_central n_bound tweb`
       (`log_Z_*` = log10 Z/Z☉; −inf when Z=0)
     - Example: `--cut log_ssfr:56:-1: --ssfr-max -2` means star-forming at
       snapshot 56 and quenched at z=0.
     - A galaxy that doesn't exist at that snapshot fails the cut.
     - A cut at a snapshot other than 127 decompresses the whole field
       (about 1 min each).
   - `--spread-by FIELD [--spread-pct 1 99]`: use this for "spread across the
     range of X". It replaces mass binning. It sets `--n` targets evenly spaced
     in FIELD between those percentiles of the galaxies passing the cuts, and
     picks the nearest galaxy to each. The result is deterministic. Output is
     sorted by FIELD (not mass), and `_selection.png` shows mass vs FIELD.
   - Look up the redshift of a snapshot in the master file's `/redshift`.
     Some reference points: 127 → z=0, 60 → z≈3.5.
   - `--seed 42`
   - With no `--select`, the defaults reproduce the original even-spread
     selection from `no_envr.ipynb`: 100 galaxies, 20 bins, 8–12, seed 42.
   - If Jake asks for a criterion that `--cut` can't express, add an option to
     `select_tracks_for_movie.py` (it's a plain .py, so editing it is fine). If
     the criterion needs a quantity the master file doesn't store, see "Master
     file". Keep the output ordered by mass.
   - If he gives explicit TrackIds, write them to a file and pass `--ids FILE`.
     They must all be in the master file (log M* > 8 at z=0).
4. **How the tracks look (`--frame-args "..."`).** These go to `make_movie_frame.py`.
   - `--n-tracks N --track-stride S`: draw a subset of the selected tracks,
     taken by position in the file (which is in mass order). Drawing 100 tracks
     at once is cluttered; Jake's usual choice is `--n-tracks 100 --track-stride 4`,
     which gives 25. If the selection is ~100 and he doesn't say, use that.
     Alternatively, select only as many galaxies as you want to draw.
   - `--tail 15`: trail length in snapshots
   - `--min-run 2`: suppresses tracks that flicker on for a single snapshot
   - `--norm fixed|auto`:
     - `fixed` (default): the colour scale is capped at the hard-coded z=0
       maximum, so you see the density build up.
     - `auto`: each frame is scaled to its own maximum. This matches the
       `*_auto` frames/movies in `animations/no_tracks/`.
   - The trail colour, marker size and the 0.08 jitter on the quenched floor
     are constants at the top of `make_movie_frame.py`.
5. **Other options:**
   - `--snaps FIRST-LAST`
   - `--array SPEC`: render only some frames (see "When a step fails")
   - `--fps 5`
   - `--tracks-npz FILE`: reuse existing histories and skip extraction
   - `--ids FILE`
   - `--no-tracks`
   - `--dry-run`

**Pass `--tracks-npz` when only the plot styling changes.** That reuses the
same galaxies, so they don't need selecting again. Existing histories are
`data/*/track_histories.npz` from earlier runs. They also contain `log_gas_mass`,
  `log_host_m200c`, `r_half_kpc`, `stellar_metallicity`, `gas_metallicity`,
  `mag_g`, `mag_r` and `tweb_z0`. The default 100 even-spread galaxies can
  always be re-extracted by running with no `--select`.

**For a quick test first**, run `--snaps 115-127` with a `_test` NAME. Note
that trails can only reach back as far as the first snapshot you included.

### Worked examples

- "Mass–sSFR movie with 25 tracks of centrals spread across mass":
  `code/submit_movie.sh mass_sSFR_centrals_even --select "--centrals central" --frame-args "--n-tracks 100 --track-stride 4"`
- "25 void galaxies vs 25 knot galaxies". Make two movies:
  `code/submit_movie.sh mass_sSFR_voids --select "--tweb 0 --n 25 --n-bins 5"` and
  `code/submit_movie.sh mass_sSFR_knots --select "--tweb 3 --n 25 --n-bins 5"`
- "Satellites in clusters (host above 10^14)":
  `code/submit_movie.sh mass_sSFR_cluster_sats --select "--centrals satellite --m200-min 14 --n 30 --n-bins 6"`
- "Galaxies that were star-forming at z~1 but are quenched now". First look up
  the snapshot nearest z=1 in `/redshift`, call it S. Then:
  `code/submit_movie.sh mass_sSFR_quenched_since_z1 --select "--cut log_ssfr:S:-1: --ssfr-max -2 --mass-min 9.5 --n 30 --n-bins 0"`
- "Mass–metallicity movie, 25 tracks at 10^11 spread across the available
  metallicities" (Jake chose stellar Z and the 1–99% range):
  `code/submit_movie.sh mass_Zstar_1e11_Zspread --plot mass-Zstar --select "--mass-min 10.9 --mass-max 11.1 --n 25 --spread-by log_Z_star"`
- "Same tracks as before but longer tails on mass–SFR":
  `code/submit_movie.sh mass_SFR_even_tail30 --plot mass-SFR --tracks-npz data/NAME_OF_EARLIER_MOVIE/track_histories.npz --frame-args "--n-tracks 100 --track-stride 4 --tail 30"`
- "Colour–magnitude, auto-scaled, no tracks":
  `code/submit_movie.sh colour_mag_auto --plot col-mag --no-tracks --frame-args "--norm auto"`

## Several movies at once (same plot, different galaxies)

If Jake asks for several movies that share a plot type and differ only in which
galaxies are tracked, use `code/submit_movie_set.sh` instead of several
`submit_movie.sh` calls. Examples: the same selection at several masses, or
voids vs knots. Each frame task then loads its snapshot once and writes one
frame per movie (`make_movie_frame.py` takes repeated
`--tracks-file F --out-dir D` pairs). That gives 126 tasks instead of 126×K,
with each SOAP catalogue read once.

1. Write a spec file in `data/SETNAME_spec.txt`, one movie per line, as
   `NAME<TAB>select args`. Lines starting with `#` and blank lines are ignored.
   Put `-` as the args for a background-only movie. It then shares the same
   frame tasks, which is how to add the "no tracks" version of a set.
2. Run `code/submit_movie_set.sh SETNAME --plot TYPE --spec FILE [--frame-args ".."] [--snaps A-B] [--fps N] [--dry-run]`.

How the set script behaves:
- It needs the master file, and only works for plot types with a track overlay
  (mass-sSFR, mass-SFR, mass-Zstar, mass-colour).
- The jobs are:
  - an `extract` array (one task per movie)
  - a `frames` array (one task per snapshot, 24G, 30 min)
  - a `stitch` array (one task per movie)
- Each movie's outputs land in the usual per-NAME places, with a
  `movie_config.txt` per movie.
- The spec is saved as `data/SETNAME_set_spec.txt`.
- Job names are `mvset_{extract,frames,stitch}_SETNAME`.
- Logs are `logs/movieset_SETNAME_*`.

Fixing failures:
- `--array 17,45` re-renders those frames for every movie, reusing the npz
  files, then re-stitches.
- `--stitch-only` only re-stitches.
- Neither needs `--spec` again.

Worked example: "the same Z*-spread movie at 11.5, 10.5, 10, 9.5, 9 and 8.5 dex".
`data/mass_Zstar_massladder_spec.txt` has lines like
`mass_Zstar_1e10.5_Zspread<TAB>--mass-min 10.4 --mass-max 10.6 --n 25 --spread-by log_Z_star`. Then:
`code/submit_movie_set.sh mass_Zstar_massladder --plot mass-Zstar --spec data/mass_Zstar_massladder_spec.txt`

Second worked example: "mass vs colour, one with no tracks and the rest at the same mass intervals".
`data/mass_colour_massladder_spec.txt` has `mass_colour<TAB>-`, then one
`mass_colour_1e{M}_grspread<TAB>--mass-min M-0.1 --mass-max M+0.1 --n 25 --spread-by g_r`
line for each M in 11.5, 11, …, 8.5. The `--plot` is `mass-colour`.

## After submitting

Tell Jake:
- the job IDs
- the movie path
- a rough ETA: extract a few minutes, frames ~10 min once
  they start, stitch minutes
- for a new selection, `data/NAME/track_IDs_selection.png`, which shows
  where the selected galaxies sit at the selection snapshot. It appears after
  step 1.

Monitoring:
- `squeue -u $USER -n mv_extract_NAME,mv_frames_NAME,mv_stitch_NAME`
- `sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS`
- Logs are in `logs/movie_NAME_{extract,frame,stitch}_*.{out,err}`.
  The `.err` files always contain harmless `divide by zero in log10` warnings
  from SFR=0 galaxies.

When a step fails:
- Read the `.err` file.
- **Frames:** list the snapshot numbers missing from the frame directory.
  Rerun only those, using the same `--plot`/`--frame-args`/`--snaps` as before:
  `code/submit_movie.sh NAME --overwrite --tracks-npz data/NAME/track_histories.npz --array 17,45,90-92 ...`
  This renders those frames and then re-stitches automatically. (For a
  background-only movie, use `--no-tracks` instead of `--tracks-npz`.) To just
  re-stitch, run `code/submit_movie.sh NAME --stitch-only`.
- **"Nothing passes the cuts" / empty bins:** loosen the cuts, or use
  `--n-bins 0`.
- **Stitch fails with `libSDL2-2.0.so.0: cannot open shared object file`:** `ffmpeg/7.0.1` links against system SDL2, which some compute nodes lack. The stitch job now loads `ffmpeg/260121_prebuild` (self-contained, has libx264). `7.0.1` is still fine on the login node for manual stitching.
- **Redshift mismatch error in frames:** the npz and the frames cover
  different snapshots. Check `--snaps`.

## Master file

Build or rebuild it with `code/submit_master_tracks.sh`; `--test --snaps 120-127`
does a small trial into `data/master_tracks_test/`. The chain is:
1. `ids` (z=0 list + T-web)
2. `parts` (array over snapshots, `build_master_tracks.py --snap N` →
   `parts/snap_NNNN.npz`)
3. `merge` (`merge_master_tracks.py` → HDF5). This deletes `parts/` only after
   a successful merge.

If some part tasks fail: `--array 17,45` rebuilds just those snapshots, and
`--merge-only` reruns the merge. The script refuses to overwrite an existing
master file, so move it aside first.

Layout:
- `/track_id`, `/z0_row`: in z=0 SOAP row order, which keeps seeded
  selections identical to the old SOAP-based ones
- `/tweb_z0`, `/centre_of_mass_z0`
- `/snap_indices`, `/redshift`
- per-snapshot `(n_snaps, n_tracks)` datasets of raw SOAP values, NaN or -1
  where the galaxy is absent: `stellar_mass`, `star_formation_rate`,
  `gas_mass`, `half_mass_radius_stars`, `stellar_metallicity`,
  `gas_metallicity`, `lum_g`, `lum_r`, `host_m200c`, `n_bound`, `is_central`,
  `present`. Units are stored as attrs.
- `/z0/<field>`: a copy of the z=0 row of each field, for fast selection

To add a field, add it to `MASTER_FLOAT_FIELDS` in `code/track_tools.py`, add
a `derived()` entry in `select_tracks_for_movie.py`, and rebuild the master
file (one array job; check `sacct` for the elapsed time).

## Limitations / where to extend

- The track overlay works for mass-sSFR, mass-SFR, mass-Zstar and mass-colour. For col-mag, the
  track npz already has `mag_g`/`mag_r`. What's missing is a
  col-mag branch in the track overlay of `make_movie_frame.py` (x = g−r,
  y = M_r, inverted y axis) and dropping the col-mag check in `submit_movie.sh`.
- A new plot type needs a new `elif plot_syntax == ...` block in
  `make_movie_frame.py`, with its own `plot_range` and fixed `vmax`, plus an
  entry in the `--plot` check in `submit_movie.sh`. Find vmax by computing the
  z=0 `hist2d` maximum, as in `simulation_envr/code/no_envr.ipynb` cell 22.
- The run (`colibre`, `L400_m7`, `Thermal`) is set once in `track_tools.SIM`.
- The T-web class only exists at z=0. Environment at higher z would need
  positions per snapshot plus a T-web grid at that snapshot.

## Manual stitch (equivalent of step 3)

```bash
source /cosma/apps/dp004/dc-howa2/venvs/jup_venv/bin/activate
module load ffmpeg/7.0.1
cd /cosma8/data/dp004/dc-howa2/colibre_movies/frames/NAME
ffmpeg -framerate 5 -start_number 2 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p \
    /cosma8/data/dp004/dc-howa2/colibre_movies/animations/NAME.mp4
```
Add `-frames:v 15` to stitch only the first 15 frames as a quick test.
