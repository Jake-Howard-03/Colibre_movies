#!/bin/bash
# Submit several movies that share one plot type, so each snapshot is loaded
# once and all of their frames are drawn from that one load.
#
# Each movie has its own galaxy selection (from the master histories file) and
# ends up exactly where code/submit_movie.sh would put it:
#   data/NAME/, frames/NAME/, animations/NAME.mp4
#
# Jobs (chained with --dependency=afterok; a failure cancels the later steps):
#   1. extract  array 1-K       select_tracks_for_movie.py, one per movie
#   2. frames   array over snaps make_movie_frame.py with one --tracks-file/--out-dir pair per movie
#   3. stitch   array 1-K       ffmpeg, one per movie
#
# Usage:
#   code/submit_movie_set.sh SETNAME --spec FILE [options]
#
# Spec file: one movie per line, NAME<TAB>args for select_tracks_for_movie.py.
# Use - as the args for a background-only movie (no track overlay) that
# shares the same frame tasks. Blank lines and lines starting with # are
# ignored. It is copied to
# data/SETNAME_set_spec.txt, which the array tasks read.
#
# Options:
#   --spec FILE          movie list (required, except with --stitch-only/--array
#                        when the saved copy already exists)
#   --plot TYPE          mass-sSFR (default) | mass-SFR | mass-Zstar | mass-colour
#   --frame-args "ARGS"  extra args for make_movie_frame.py (same for every movie)
#   --snaps FIRST-LAST   snapshot range (default 2-127)
#   --array SPEC         only re-render these frames (reuses existing npz files;
#                        implies --overwrite, skips extract), then re-stitch all
#   --fps N              movie frame rate (default 5)
#   --overwrite          allow reusing existing NAMEs / SETNAME
#   --stitch-only        only re-stitch every movie in the set
#   --dry-run            print the job scripts, submit nothing
#
# Example:
#   code/submit_movie_set.sh mass_Zstar_massladder --plot mass-Zstar \
#       --spec data/mass_Zstar_massladder_spec.txt

set -euo pipefail

ROOT=/cosma8/data/dp004/dc-howa2/colibre_movies
ENVR_CODE=/cosma8/data/dp004/dc-howa2/simulation_envr/code  # SWIFT_tools, Tweb_tools, plotting_funcs
VENV=/cosma/apps/dp004/dc-howa2/venvs/jup_venv/bin/activate
LOGDIR=$ROOT/logs
MASTER=$ROOT/data/master_tracks/track_histories_logM8.hdf5
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
SET=$1; shift
[[ $SET =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "SETNAME must be [A-Za-z0-9_.-]" >&2; exit 1; }

PLOT=mass-sSFR; SPEC=""; FRAME_ARGS=""; SNAPS=2-127; FPS=5
OVERWRITE=0; DRY=0; STITCH_ONLY=0; ARRAY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --spec) SPEC=$(realpath "$2"); shift 2 ;;
        --plot) PLOT=$2; shift 2 ;;
        --frame-args) FRAME_ARGS=$2; shift 2 ;;
        --snaps) SNAPS=$2; shift 2 ;;
        --array) ARRAY=$2; OVERWRITE=1; shift 2 ;;
        --fps) FPS=$2; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --stitch-only) STITCH_ONLY=1; OVERWRITE=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

case $PLOT in mass-sSFR|mass-SFR|mass-Zstar|mass-colour) ;;
    *) echo "Bad --plot $PLOT (col-mag has no track overlay)" >&2; exit 1 ;; esac
[[ -e $MASTER ]] || { echo "Master file $MASTER not found; build it with code/submit_master_tracks.sh" >&2; exit 1; }

SNAP_FIRST=${SNAPS%-*}; SNAP_LAST=${SNAPS#*-}
REDO_FRAMES=0; [[ -n $ARRAY ]] && REDO_FRAMES=1
[[ -n $ARRAY ]] || ARRAY=$SNAP_FIRST-$SNAP_LAST

SAVED_SPEC=$ROOT/data/${SET}_set_spec.txt
if [[ -n $SPEC && $SPEC != "$SAVED_SPEC" ]]; then
    if [[ -e $SAVED_SPEC && $OVERWRITE == 0 ]]; then
        echo "Set '$SET' already exists ($SAVED_SPEC). Pick a new SETNAME or pass --overwrite." >&2
        exit 1
    fi
    SPEC_LINES=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$SPEC")
elif [[ -e $SAVED_SPEC ]]; then
    SPEC_LINES=$(cat "$SAVED_SPEC")
else
    echo "--spec FILE is required" >&2; exit 1
fi

NAMES=(); HAS_TRACKS=()
while IFS=$'\t' read -r name sel; do
    [[ $name =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Bad movie NAME '$name' in spec (needs NAME<TAB>args)" >&2; exit 1; }
    NAMES+=("$name")
    if [[ -z ${sel// } || $sel == - ]]; then HAS_TRACKS+=(0); else HAS_TRACKS+=(1); fi
done <<< "$SPEC_LINES"
K=${#NAMES[@]}
[[ $(printf '%s\n' "${NAMES[@]}" | sort | uniq -d) == "" ]] || { echo "Duplicate NAME in spec" >&2; exit 1; }

frame_dir() { echo "$ROOT/frames/$1"; }

if [[ $OVERWRITE == 0 ]]; then
    for n in "${NAMES[@]}"; do
        if [[ -e $ROOT/data/$n || -e $(frame_dir "$n") || -e $ROOT/animations/$n.mp4 ]]; then
            echo "Movie '$n' already exists. Rename it in the spec or pass --overwrite." >&2; exit 1
        fi
    done
fi
if [[ $STITCH_ONLY == 1 || $REDO_FRAMES == 1 ]]; then
    for i in "${!NAMES[@]}"; do
        n=${NAMES[$i]}
        [[ ${HAS_TRACKS[$i]} == 0 || -e $ROOT/data/$n/track_histories.npz ]] || { echo "No track histories for $n" >&2; exit 1; }
    done
fi

if [[ $DRY == 0 ]]; then
    mkdir -p "$LOGDIR" "$ROOT/animations"
    printf '%s\n' "$SPEC_LINES" > "$SAVED_SPEC"
    for n in "${NAMES[@]}"; do mkdir -p "$ROOT/data/$n" "$(frame_dir "$n")"; done
fi

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

# Bash snippet run inside array jobs: NAME and SEL for movie $SLURM_ARRAY_TASK_ID
PICK_LINE="IFS=\$'\\t' read -r NAME SEL < <(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $SAVED_SPEC)"

LAST=""

# --- 1. extract ---------------------------------------------------------------
if [[ $STITCH_ONLY == 0 && $REDO_FRAMES == 0 ]]; then
LAST=$(submit "#!/bin/bash -l
#SBATCH -J mvset_extract_$SET
$SBATCH_COMMON
#SBATCH --array=1-$K
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o $LOGDIR/movieset_${SET}_extract_%A_%a.out
#SBATCH -e $LOGDIR/movieset_${SET}_extract_%A_%a.err
$ENV_SETUP
$PICK_LINE
echo \"movie \$NAME: \$SEL\"
if [[ -z \${SEL// } || \$SEL == - ]]; then echo 'background only, nothing to extract'; exit 0; fi
D=$ROOT/data/\$NAME
python -u select_tracks_for_movie.py --out \$D/track_IDs.txt \\
    --tracks-out \$D/track_histories.npz \$SEL")
echo "extract: $LAST (array 1-$K)"
fi

# --- 2. frames ------------------------------------------------------------------
PAIRS=""
for i in "${!NAMES[@]}"; do
    n=${NAMES[$i]}
    tf=$ROOT/data/$n/track_histories.npz
    [[ ${HAS_TRACKS[$i]} == 1 ]] || tf=none
    PAIRS+=" --tracks-file $tf --out-dir $(frame_dir "$n")"
done
if [[ $STITCH_ONLY == 0 ]]; then
LAST=$(submit "#!/bin/bash -l
#SBATCH -J mvset_frames_$SET
$SBATCH_COMMON
#SBATCH --array=$ARRAY
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o $LOGDIR/movieset_${SET}_frame_%A_%a.out
#SBATCH -e $LOGDIR/movieset_${SET}_frame_%A_%a.err
$ENV_SETUP
python -u make_movie_frame.py \${SLURM_ARRAY_TASK_ID} --plot-syntax $PLOT --tracks \\
   $PAIRS $FRAME_ARGS" "$LAST")
echo "frames: $LAST (array $ARRAY, $K frames per task)"
fi

# --- 3. stitch -------------------------------------------------------------------
LAST=$(submit "#!/bin/bash -l
#SBATCH -J mvset_stitch_$SET
$SBATCH_COMMON
#SBATCH --array=1-$K
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o $LOGDIR/movieset_${SET}_stitch_%A_%a.out
#SBATCH -e $LOGDIR/movieset_${SET}_stitch_%A_%a.err
module purge
# ffmpeg/7.0.1 needs libSDL2, which some compute nodes lack; the prebuild is self-contained
module load ffmpeg/260121_prebuild
$PICK_LINE
cd $ROOT/frames/\$NAME
ffmpeg -y -framerate $FPS -start_number $SNAP_FIRST -i frame_%04d.png \\
    -c:v libx264 -pix_fmt yuv420p $ROOT/animations/\$NAME.mp4" "$LAST")
echo "stitch: $LAST (array 1-$K)"

# --- record what was asked for ---------------------------------------------------
if [[ $DRY == 0 && $STITCH_ONLY == 0 && $REDO_FRAMES == 0 ]]; then
    while IFS=$'\t' read -r n sel; do
        {
            echo "date: $(date -Iseconds)"
            echo "set: $SET ($SAVED_SPEC)"
            t=1; [[ -z ${sel// } || $sel == - ]] && t=0
            echo "plot: $PLOT  tracks: $t  snaps: $SNAPS  fps: $FPS  master: 1"
            echo "select args: $sel"
            echo "ids: $ROOT/data/$n/track_IDs.txt"
            echo "tracks npz: $ROOT/data/$n/track_histories.npz"
            echo "frame args: $FRAME_ARGS"
            echo "frames: $(frame_dir "$n")"
            echo "movie: $ROOT/animations/$n.mp4"
            echo "last job (stitch): $LAST"
        } > "$ROOT/data/$n/movie_config.txt"
    done <<< "$SPEC_LINES"
fi
echo "Movies will be written to:"
for n in "${NAMES[@]}"; do echo "  $ROOT/animations/$n.mp4"; done
