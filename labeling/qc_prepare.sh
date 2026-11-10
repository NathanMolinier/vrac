#!/bin/bash

# Prepare a QC folder for every sample used by discs-labeling-multicontrast
# (train / valid / test of the config's data_json + evaluation.json):
#
#   <qc_dir>/images/       staged input images      (<dataset>__<stem>.nii.gz)
#   <qc_dir>/labels/       ground-truth labels copy (same stems)
#   <qc_dir>/predictions/  model predictions        (<stem>_discs.nii.gz)
#   <qc_dir>/metrics/previews/<stem>.png   GT | prediction side by side, same
#                          colours as the evaluation `errors/` folder
#   <qc_dir>/qc_index.json where each sample comes from (BIDS paths, JSON/split)
#
# Then review with:  python qc_review.py <qc_dir>
# And apply with:    python qc_apply.py  <qc_dir>
#
# Usage:
#   qc_prepare.sh <pipeline-config.json> <qc_dir> [2d|3d]
#
# The config is the same one given to run_pipeline.sh; the model is
#   $DISCSLABEL_DATA/weights/<run_name>/best_model.pt
#
# Environment:
#   DISCSLABEL_DATA   root folder that holds `bids/` and `weights/`  (required)
#   DISCSLABEL_REPO   discs-labeling-multicontrast checkout (default: the
#                     location of the installed `discslabel` package)
#   QC_SKELETON       off | flag | correct  (default: config inference_skeleton, else off)
#   QC_STEPS          steps to run, default "stage inference evaluate"

set -e
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

if [ -z "$DISCSLABEL_DATA" ]; then
    echo "Please set the DISCSLABEL_DATA environment variable."
    exit 1
fi

config_json="$1"
QC_DIR="$2"
DIM="${3:-2d}"
if [ -z "$config_json" ] || [ ! -f "$config_json" ] || [ -z "$QC_DIR" ]; then
    echo "Usage: $0 <pipeline-config.json> <qc_dir> [2d|3d]"
    exit 1
fi
config_json="$(realpath "$config_json")"
mkdir -p "$QC_DIR"
QC_DIR="$(realpath "$QC_DIR")"
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
QC_STEPS="${QC_STEPS:-stage inference evaluate}"

if [ -z "$DISCSLABEL_REPO" ]; then
    DISCSLABEL_REPO=$(python -c "import discslabel, os; print(os.path.dirname(list(discslabel.__path__)[0]))")
fi
DATASETS_DIR="$DISCSLABEL_REPO/discslabel/datasets"

jreq() {
    local val
    val=$(jq -r "$1 // empty" "$config_json")
    if [ -z "$val" ]; then
        echo "config: $1 is required"
        exit 1
    fi
    printf '%s' "$val"
}
jopt() { jq -r "if $1 == null then $2 else $1 end" "$config_json"; }

# -------- Read the config (same defaults as run_pipeline.sh / run_pipeline_3d.sh) --------
DATA_JSON=$(jreq '.data_json')
OUT_NAME=$(jopt  '.out_name' '"'$(basename "${DATA_JSON%.json}")'"')
if [ "$DIM" = "3d" ]; then
    RUN_NAME=$(jopt '.run_name' '"'$OUT_NAME'_UNet3D"')
    INFERENCE_CMD=discslabel_inference_3d
else
    RUN_NAME=$(jopt '.run_name' '"'$OUT_NAME'_UNet"')
    INFERENCE_CMD=discslabel_inference
fi
SKELETON="${QC_SKELETON:-$(jopt '.inference_skeleton' '"off"')}"
TOLERANCE=$(jopt '.val_tolerance_mm' '10')

if [ ! -f "$DATA_JSON" ]; then
    DATA_JSON="$DATASETS_DIR/$(basename "${DATA_JSON%.json}").json"
fi
EVAL_JSON="$DATASETS_DIR/evaluation.json"
for f in "$DATA_JSON" "$EVAL_JSON"; do
    if [ ! -f "$f" ]; then echo "Error: missing $f (set DISCSLABEL_REPO?)"; exit 1; fi
done
DATA_JSON="$(realpath "$DATA_JSON")"
EVAL_JSON="$(realpath "$EVAL_JSON")"

WEIGHTS="$DISCSLABEL_DATA/weights/$RUN_NAME/best_model.pt"
BIDS_ROOT="$DISCSLABEL_DATA/bids"

echo ""
echo "Config:    $config_json  ($DIM)"
echo "Weights:   $WEIGHTS"
echo "Data JSON: $DATA_JSON"
echo "Eval JSON: $EVAL_JSON"
echo "QC dir:    $QC_DIR"
echo "Steps:     $QC_STEPS  (skeleton=$SKELETON, tolerance=${TOLERANCE}mm)"
echo ""

cp "$config_json" "$QC_DIR/config.json"

# -------- 1. Stage images + GT labels --------
if [[ " $QC_STEPS " == *" stage "* ]]; then
    echo "==> staging images and ground truth"
    python "$SCRIPT_DIR/qc_stage.py" \
        --bids-root "$BIDS_ROOT" \
        --qc-dir "$QC_DIR" \
        --data-json "$DATA_JSON" "$EVAL_JSON"
fi

# -------- 2. Inference --------
if [[ " $QC_STEPS " == *" inference "* ]]; then
    if [ ! -f "$WEIGHTS" ]; then echo "Error: $WEIGHTS not found"; exit 1; fi
    echo "==> $INFERENCE_CMD"
    "$INFERENCE_CMD" \
        --weights "$WEIGHTS" \
        --input "$QC_DIR/images" \
        --out-dir "$QC_DIR/predictions" \
        --config "$config_json" \
        --skeleton "$SKELETON"
fi

# -------- 3. GT | prediction previews for every image --------
if [[ " $QC_STEPS " == *" evaluate "* ]]; then
    echo "==> discslabel_evaluate (previews for all images)"
    discslabel_evaluate \
        --gt-dir "$QC_DIR/labels" \
        --image-dir "$QC_DIR/images" \
        --pred-dir "$QC_DIR/predictions" \
        --out-dir "$QC_DIR/metrics" \
        --run-name "qc_$RUN_NAME" \
        --tolerance-mm "$TOLERANCE" \
        --previews all
fi

echo ""
echo "QC folder ready: $QC_DIR"
echo "Review:  python $SCRIPT_DIR/qc_review.py $QC_DIR"
echo "Apply:   python $SCRIPT_DIR/qc_apply.py $QC_DIR --dry-run"
