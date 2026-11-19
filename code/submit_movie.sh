#!/bin/bash
# Submit the full movie pipeline as a chain of Slurm jobs.
#
# Needs the master histories file (data/master_tracks/, built once by
# code/submit_master_tracks.sh).
#   1. extract select_tracks_for_movie.py   -> data/NAME/track_IDs.txt
#                                              + data/NAME/track_histories.npz
#   2. frames  make_movie_frame.py (array)  -> frames/NAME/frame_XXXX.png
#   3. stitch  ffmpeg                       -> animations/NAME.mp4
#
# Each step waits for the previous one with --dependency=afterok. If any step
# (or any single frame) fails, Slurm cancels the later steps automatically.
# Re-run the failed frames by hand, then use --stitch-only.
#
# Usage:
#   code/submit_movie.sh NAME [options]
#
# Options:
#   --plot TYPE          mass-sSFR (default) | mass-SFR | mass-Zstar | mass-colour | col-mag
#   --no-tracks          background only: skip step 1, no overlay
#   --select "ARGS"      args for select_tracks_for_movie.py (default: its defaults)
#   --ids FILE           use an existing TrackId file instead of selecting
#   --tracks-npz FILE    reuse existing track histories, skip step 1
#   --frame-args "ARGS"  extra args for make_movie_frame.py
#                        (e.g. "--n-tracks 100 --track-stride 4 --tail 15 --norm auto")
#   --snaps FIRST-LAST   snapshot range (default 2-127)
#   --array SPEC         only render these frames, e.g. "17,45,90-92" (for redoing
#                        failed frames; combine with --overwrite --tracks-npz)
#   --fps N              movie frame rate (default 5)
#   --overwrite          allow reusing an existing NAME
#   --stitch-only        only (re)run step 3 on existing frames for NAME
#   --dry-run            print the job scripts, submit nothing
#
# Example:
#   code/submit_movie.sh sSFR_centrals_even --select "--centrals central --n 50 --n-bins 10" \
#       --frame-args "--tail 20"

set -euo pipefail

ROOT=/cosma8/data/dp004/dc-howa2/colibre_movies
ENVR_CODE=/cosma8/data/dp004/dc-howa2/simulation_envr/code  # SWIFT_tools, Tweb_tools, plotting_funcs
VENV=/cosma/apps/dp004/dc-howa2/venvs/jup_venv/bin/activate
LOGDIR=$ROOT/logs
SBATCH_COMMON="#SBATCH -A dp004
#SBATCH -p cosma8-serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1"
ENV_SETUP="module purge
module load cosma/2024
source $VENV
export PYTHONPATH=$ENVR_CODE
cd $ROOT/code"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
[[ $1 == -h || $1 == --help ]] && usage 0
NAME=$1; shift
[[ $NAME =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "NAME must be [A-Za-z0-9_.-]" >&2; exit 1; }

PLOT=mass-sSFR; TRACKS=1; SELECT_ARGS=""; IDS=""; NPZ=""; FRAME_ARGS=""
SNAPS=2-127; FPS=5; OVERWRITE=0; DRY=0; STITCH_ONLY=0; ARRAY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --plot) PLOT=$2; shift 2 ;;
        --no-tracks) TRACKS=0; shift ;;
        --select) SELECT_ARGS=$2; shift 2 ;;
        --ids) IDS=$(realpath "$2"); shift 2 ;;
        --tracks-npz) NPZ=$(realpath "$2"); shift 2 ;;
        --frame-args) FRAME_ARGS=$2; shift 2 ;;
        --snaps) SNAPS=$2; shift 2 ;;
        --array) ARRAY=$2; shift 2 ;;
        --fps) FPS=$2; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --stitch-only) STITCH_ONLY=1; OVERWRITE=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

case $PLOT in mass-sSFR|mass-SFR|mass-Zstar|mass-colour|col-mag) ;; *) echo "Bad --plot $PLOT" >&2; exit 1 ;; esac
if [[ $PLOT == col-mag && $TRACKS == 1 && $STITCH_ONLY == 0 ]]; then
    echo "Track overlay is not implemented for col-mag; add --no-tracks" >&2; exit 1
fi
SNAP_FIRST=${SNAPS%-*}; SNAP_LAST=${SNAPS#*-}
[[ -n $ARRAY ]] || ARRAY=$SNAP_FIRST-$SNAP_LAST

RUN_DIR=$ROOT/data/$NAME
FRAME_DIR=$ROOT/frames/$NAME
MOVIE=$ROOT/animations/$NAME.mp4

if [[ $OVERWRITE == 0 && ( -e $RUN_DIR || -e $FRAME_DIR || -e $MOVIE ) ]]; then
    echo "Movie '$NAME' already exists (one of $RUN_DIR, $FRAME_DIR, $MOVIE)." >&2
    echo "Pick a new NAME or pass --overwrite." >&2
    exit 1
fi

[[ $DRY == 1 ]] || mkdir -p "$RUN_DIR" "$FRAME_DIR" "$LOGDIR" "$ROOT/animations"

# submit SCRIPT [DEPENDENCY_JOBID] -> echoes the job id
submit() {
    local script=$1 dep=${2:-}
    local depflag=()
    [[ -n $dep ]] && depflag=(--dependency=afterok:$dep --kill-on-invalid-dep=yes)
    if [[ $DRY == 1 ]]; then
        printf '\n----- job (after %s) -----\n%s\n' "${dep:-none}" "$script" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "${depflag[@]}" <<< "$script"
    fi
}

LAST=""

if [[ $STITCH_ONLY == 1 ]]; then
    [[ -d $FRAME_DIR ]] || { echo "No frames at $FRAME_DIR" >&2; exit 1; }
    TRACKS=0
fi

MASTER=$ROOT/data/master_tracks/track_histories_logM8.hdf5

# --- 1. extract from master -------------------------------------------------
if [[ $STITCH_ONLY == 0 && $TRACKS == 1 && -z $NPZ ]]; then
    [[ -e $MASTER ]] || { echo "No master file at $MASTER; build it with code/submit_master_tracks.sh" >&2; exit 1; }
    NPZ=$RUN_DIR/track_histories.npz
    IDFLAG=""; [[ -n $IDS ]] && IDFLAG="--ids $IDS"
    LAST=$(submit "#!/bin/bash -l
#SBATCH -J mv_extract_$NAME
$SBATCH_COMMON
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o $LOGDIR/movie_${NAME}_extract_%j.out
#SBATCH -e $LOGDIR/movie_${NAME}_extract_%j.err
$ENV_SETUP
python -u select_tracks_for_movie.py --out $RUN_DIR/track_IDs.txt \
    --tracks-out $NPZ $IDFLAG $SELECT_ARGS")
    echo "extract: $LAST"
    IDS=$RUN_DIR/track_IDs.txt
fi

# --- 2. frames ---------------------------------------------------------------
TRACK_FLAGS=""
[[ $TRACKS == 1 ]] && TRACK_FLAGS="--tracks --tracks-file $NPZ"
if [[ $STITCH_ONLY == 0 ]]; then
LAST=$(submit "#!/bin/bash -l
#SBATCH -J mv_frames_$NAME
$SBATCH_COMMON
#SBATCH --array=$ARRAY
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o $LOGDIR/movie_${NAME}_frame_%A_%a.out
#SBATCH -e $LOGDIR/movie_${NAME}_frame_%A_%a.err
$ENV_SETUP
python -u make_movie_frame.py \${SLURM_ARRAY_TASK_ID} --plot-syntax $PLOT \\
    --out-dir $FRAME_DIR $TRACK_FLAGS $FRAME_ARGS" "$LAST")
echo "frames: $LAST (array $ARRAY)"
fi

# --- 3. stitch ---------------------------------------------------------------
LAST=$(submit "#!/bin/bash -l
#SBATCH -J mv_stitch_$NAME
$SBATCH_COMMON
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o $LOGDIR/movie_${NAME}_stitch_%j.out
#SBATCH -e $LOGDIR/movie_${NAME}_stitch_%j.err
module purge
# ffmpeg/7.0.1 needs libSDL2, which some compute nodes lack; the prebuild is self-contained
module load ffmpeg/260121_prebuild
cd $FRAME_DIR
ffmpeg -y -framerate $FPS -start_number $SNAP_FIRST -i frame_%04d.png \\
    -c:v libx264 -pix_fmt yuv420p $MOVIE" "$LAST")
echo "stitch: $LAST"

# --- record what was asked for ------------------------------------------------
if [[ $DRY == 0 && $STITCH_ONLY == 0 ]]; then
    {
        echo "date: $(date -Iseconds)"
        echo "plot: $PLOT  tracks: $TRACKS  snaps: $SNAPS  fps: $FPS"
        echo "select args: $SELECT_ARGS"
        echo "ids: $IDS"
        echo "tracks npz: $NPZ"
        echo "frame args: $FRAME_ARGS"
        echo "frames: $FRAME_DIR"
        echo "movie: $MOVIE"
        echo "last job (stitch): $LAST"
    } > "$RUN_DIR/movie_config.txt"
fi
echo "Movie will be written to $MOVIE"
