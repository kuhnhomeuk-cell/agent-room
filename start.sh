#!/bin/sh
# Start the Agent Room (if it is not already running) and open it in the browser.
# The agents work in AGENT_ROOM_WORKDIR, or in the folder you run this from.
ROOM="$(cd "$(dirname "$0")" && pwd)"
URL="http://127.0.0.1:8787"
mkdir -p "$ROOM/logs"

up() { curl -s -o /dev/null --max-time 1 "$URL/api/messages"; }

if up; then
  echo "Agent Room was already running, so its working folder is unchanged."
  echo "To switch folders, stop server.py and run this again."
else
  nohup python3 "$ROOM/server.py" >>"$ROOM/logs/server.log" 2>&1 &
  i=0
  while [ $i -lt 12 ] && ! up; do sleep 0.5; i=$((i+1)); done
fi

if up; then
  echo "Agent Room is running at $URL"
  (open "$URL" || xdg-open "$URL") >/dev/null 2>&1
else
  echo "The Agent Room server failed to start. Last lines of logs/server.log:"
  tail -20 "$ROOM/logs/server.log"
  exit 1
fi
