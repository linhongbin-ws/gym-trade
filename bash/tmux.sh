#!/usr/bin/env bash

SESSION_NAME="gym_trade"

# Everything after the script name is treated as the command
CMD="$*"

if [[ -z "$CMD" ]]; then
  echo "Usage: $0 <command>"
  echo "Example:"
  echo "  $0 \"source ./tmux_bash.sh && python test file\""
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" bash -lc "$CMD"
