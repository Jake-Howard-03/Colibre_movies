# colibre_movies: redshift-evolution movies of COLIBRE galaxy populations

Split out of `/cosma8/data/dp004/dc-howa2/simulation_envr` in Nov 2026. That
repo holds the science (galaxy properties vs cosmic-web environment); this one
only makes animations: a 2D density plot (mass–sSFR, mass–SFR, mass–Z*,
mass–colour, colour–magnitude) evolving from snapshot 2 to 127 (z=0), with
optional trails of selected galaxies.

To make a movie, use the `make-movie` skill (`.claude/skills/make-movie/SKILL.md`).
It explains the whole pipeline and how to turn a request into one
`code/submit_movie.sh` call.

## Layout
- `code/`
  - `track_tools.py`: paths, constants (`SIM`, `APERTURE`, `H`, `Z_SUN`,
    `SSFR_FLOOR`), master-file field list, and shared post-processing. **Every
    script imports it first**, because it adds `simulation_envr/code` to
    `sys.path` for `SWIFT_tools`, `Tweb_tools` and `plotting_funcs`.
  - `build_master_tracks.py`, `merge_master_tracks.py`, `submit_master_tracks.sh`:
    build the one-off master history file of all log M* > 8 (z=0) galaxies.
  - `select_tracks_for_movie.py`: pick galaxies from the master file and write
    their histories.
  - `make_movie_frame.py`: one frame per snapshot (Slurm array task).
  - `submit_movie.sh` (one movie), `submit_movie_set.sh` (several movies sharing
    the snapshot loads): chain extract → frames → stitch as Slurm jobs.
- `data/`: `master_tracks/` (~3.5 GB HDF5) and `NAME/` per movie (track IDs,
  histories npz, selection plot, `movie_config.txt`), plus set spec files.
- `frames/NAME/frame_XXXX.png`, `animations/NAME.mp4`, `logs/` (Slurm output).
- `data/`, `frames/`, `animations/`, `logs/` are gitignored.

Movies made before the move have `movie_config.txt` / `_selection.json` files
that still point at the old `simulation_envr/...` locations. Those records are
historical; the files themselves now live here.

## Environment
- Python venv: `/cosma/apps/dp004/dc-howa2/venvs/jup_venv` with `module load cosma/2024`.
- Simulation: COLIBRE `L400_m7` / `Thermal`, SOAP-HBT catalogues, aperture
  `exclusive_sphere_50kpc`.
- Changes to `SWIFT_tools.py` / `Tweb_tools.py` in simulation_envr affect this
  project too.
- Jake has authorised submitting movie jobs without asking (see the skill).
- Don't scan the `/cosma*` or `/snap*` file systems broadly.
