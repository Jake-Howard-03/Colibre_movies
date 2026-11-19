#!/bin/bash
# Build the master track histories (all galaxies with log M* > 8 at z=0,
# followed back through every snapshot) as a chain of Slurm jobs:
#
#   1. ids    build_master_tracks.py --make-ids   -> MASTER_DIR/track_ids_z0.npz
#   2. parts  build_master_tracks.py --snap N     -> MASTER_DIR/parts/snap_NNNN.npz  (array)
#   3. merge  merge_master_tracks.py              -> MASTER_DIR/track_histories_logM8.hdf5
#
# Usage:
#   code/submit_master_tracks.sh [options]
#
# Options:
#   --test               write to data/master_tracks_test/ instead (use with --snaps)
#   --snaps FIRST-LAST   snapshot range (default 2-127; must end at 127)
#   --array SPEC         only (re)build these snapshots, e.g. "17,45"; skips step 1
#   --merge-only         only run step 3
#   --keep-parts         don't delete parts/ after a successful merge
#   --dry-run            print the job scripts, submit nothing

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

DIR=$ROOT/data/master_tracks; TAG=""
SNAPS=2-127; ARRAY=""; MERGE_ONLY=0; KEEP=0; DRY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --test) DIR=$ROOT/data/master_tracks_test; TAG=_test; shift ;;
        --snaps) SNAPS=$2; shift 2 ;;
        --array) ARRAY=$2; shift 2 ;;
        --merge-only) MERGE_ONLY=1; shift ;;
        --keep-parts) KEEP=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done
SNAP_FIRST=${SNAPS%-*}; SNAP_LAST=${SNAPS#*-}
[[ $SNAP_LAST == 127 ]] || { echo "--snaps must end at 127 (z=0 selection)" >&2; exit 1; }

IDS=$DIR/track_ids_z0.npz
PARTS=$DIR/parts
OUT=$DIR/track_histories_logM8.hdf5
PATHS="--ids-file $IDS --parts-dir $PARTS"
DEL="--delete-parts"; [[ $KEEP == 1 ]] && DEL=""

if [[ -e $OUT && $MERGE_ONLY == 0 && -z $ARRAY ]]; then
    echo "$OUT already exists. Move it aside first, or use --array / --merge-only." >&2
    exit 1
fi
[[ $DRY == 1 ]] || mkdir -p "$DIR" "$LOGDIR"

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
if [[ $MERGE_ONLY == 0 ]]; then
    if [[ -z $ARRAY ]]; then
        LAST=$(submit "#!/bin/bash -l
#SBATCH -J mt_ids$TAG
$SBATCH_COMMON
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o $LOGDIR/master_tracks${TAG}_ids_%j.out
#SBATCH -e $LOGDIR/master_tracks${TAG}_ids_%j.err
$ENV_SETUP
python -u build_master_tracks.py --make-ids $PATHS")
        echo "ids:   $LAST"
        ARRAY=$SNAP_FIRST-$SNAP_LAST
    else
        [[ $DRY == 1 || -e $IDS ]] || { echo "No $IDS; run without --array first" >&2; exit 1; }
    fi

    LAST=$(submit "#!/bin/bash -l
#SBATCH -J mt_parts$TAG
$SBATCH_COMMON
#SBATCH --array=$ARRAY
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o $LOGDIR/master_tracks${TAG}_part_%A_%a.out
#SBATCH -e $LOGDIR/master_tracks${TAG}_part_%A_%a.err
$ENV_SETUP
python -u build_master_tracks.py --snap \${SLURM_ARRAY_TASK_ID} $PATHS" "$LAST")
    echo "parts: $LAST (array $ARRAY)"
fi

LAST=$(submit "#!/bin/bash -l
#SBATCH -J mt_merge$TAG
$SBATCH_COMMON
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o $LOGDIR/master_tracks${TAG}_merge_%j.out
#SBATCH -e $LOGDIR/master_tracks${TAG}_merge_%j.err
$ENV_SETUP
python -u merge_master_tracks.py $PATHS --output $OUT \\
    --snap-first $SNAP_FIRST --snap-last $SNAP_LAST $DEL" "$LAST")
echo "merge: $LAST"
echo "Master file will be $OUT"
