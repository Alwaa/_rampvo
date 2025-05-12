#!/usr/bin/env bash
set -euo pipefail

REMOTE=alexander@snaga.ifi.uzh.ch
REMOTE_ROOT=/data/storage/datasets/TartanEvent/ocean
LOCAL_ROOT='/run/media/alexander/T5 EVO/datasets/ocean'

# Cleanup function: kill any subprocesses (ssh, scp, find) on Ctrl-C
cleanup() {
  echo -e "\nInterrupted—cleaning up…"
  pkill -P $$ 2>/dev/null || true
  exit 1
}
trap cleanup INT



# 0) Count how many dirs and files we’ll process
echo "Counting remote directories and files…"
total_dirs=$(ssh "$REMOTE" "find '$REMOTE_ROOT' -type d" | wc -l)
total_files=$(ssh "$REMOTE" "find '$REMOTE_ROOT' -type f" | wc -l)
echo "→ $total_dirs directories, $total_files files to process."



# 1. Recreate directory tree
ssh "$REMOTE" "find '$REMOTE_ROOT' -type d" | \
while read -r dir; do
  mkdir -p "$LOCAL_ROOT${dir#$REMOTE_ROOT}"
done

echo "Checking and copying files…"
file_count=0
ssh "$REMOTE" "find '$REMOTE_ROOT' -type f" | \
while read -r file; do
  file_count=$((file_count + 1))
  rel="${file#$REMOTE_ROOT}"
  localfile="$LOCAL_ROOT$rel"
  printf "[%d/%d] Checking '%s'…\n" "$file_count" "$total_files" "$localfile"

  if [[ ! -e "$localfile" ]]; then
    echo "    → Copying: scp -p '$REMOTE:$file' '$localfile'"
    scp -p "$REMOTE":"$file" "$localfile"
    sleep 1
  else
    echo "    → Skipped (already exists)"
  fi

done

echo "All done!"
