#!/usr/bin/env bash
# keep-range.sh — delete images outside [START, END] by numeric basename (e.g., 00001234.jpg)
# Dry-run by default. Pass --apply to actually delete.

# set -euo pipefail

# usage() {
#   cat <<'EOF'
# Usage:
#   keep-range.sh -d DIR -s START -e END [-x "jpg,png,jpeg,webp"] [--apply]

# Description:
#   Keeps files whose basename is digits (leading zeros ok) within [START, END],
#   deletes all other matching images in DIR.

# Options:
#   -d DIR       Directory containing images
#   -s START     Start number (inclusive), decimal
#   -e END       End number (inclusive), decimal
#   -x EXTLIST   Comma-separated extensions (default: jpg,jpeg,png,webp,bmp,tif,tiff)
#   --apply      Perform deletion (otherwise just prints what would be deleted)
#   -h, --help   Show this help

# Examples:
#   # Keep 1000..2000, delete others (preview only):
#   keep-range.sh -d /path/to/frames -s 1000 -e 2000

#   # Actually delete:
#   keep-range.sh -d /path/to/frames -s 1000 -e 2000 --apply

#   # Custom extensions:
#   keep-range.sh -d ./imgs -s 42 -e 123 -x "png,jpg" --apply
# EOF
# }

dir=""
start=""
end=""
apply=0
exts="jpg,jpeg,png,webp,bmp,tif,tiff,lms"

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# --- Parse args ---
while (($#)); do
  case "$1" in
    -d)   dir="${2:-}"; shift 2 ;;
    -s)   start="${2:-}"; shift 2 ;;
    -e)   end="${2:-}"; shift 2 ;;
    -x)   exts="${2:-}"; shift 2 ;;
    --apply) apply=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# --- Validate ---
# [[ -d "$dir" ]] || { echo "ERROR: Directory not found: $dir" >&2; exit 1; }
# is_int "$start" || { echo "ERROR: START must be integer" >&2; exit 1; }
# is_int "$end"   || { echo "ERROR: END must be integer" >&2; exit 1; }
# (( start <= end )) || { echo "ERROR: START must be <= END" >&2; exit 1; }

IFS=',' read -r -a EXT_ARR <<<"$exts"
shopt -s nullglob nocaseglob

declare -a to_delete kept skipped
for ext in "${EXT_ARR[@]}"; do
  for f in "$dir"/*."$ext"; do
    base="${f##*/}"
    name="${base%.*}"

    # Only process strictly numeric basenames (e.g., 00000001)
    if [[ "$name" =~ ^[0-9]+$ ]]; then
      # Force base-10 even with leading zeros
      num=$((10#$name))
      if (( num < start || num > end )); then
        to_delete+=("$f")
      else
        kept+=("$f")
      fi
    else
      skipped+=("$f")  # not numeric; we leave it alone
    fi
  done
done

echo "Target dir      : $dir"
echo "Keep range      : [$start, $end]"
echo "Extensions      : ${EXT_ARR[*]}"
echo "Kept (in range) : ${#kept[@]}"
echo "Delete (out)    : ${#to_delete[@]}"
echo "Skipped (non-numeric basenames or unmatched ext): ${#skipped[@]}"

if (( ${#to_delete[@]} == 0 )); then
  echo "Nothing to delete."
  exit 0
fi

echo
if (( apply == 0 )); then
  echo "DRY-RUN: would delete ${#to_delete[@]} files. Listing first 20:"
  printf '  %s\n' "${to_delete[@]:0:20}"
  echo
  echo "Add --apply to actually delete."
else
  echo "Deleting ${#to_delete[@]} files..."
  # Delete in batches to avoid 'argument list too long'
  chunk=200
  i=0
  while (( i < ${#to_delete[@]} )); do
    end_idx=$(( i + chunk ))
    rm -v -- "${to_delete[@]:i:chunk}"
    i=$end_idx
  done
  echo "Done."
fi
