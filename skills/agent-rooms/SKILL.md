---
name: agent-rooms
description: Put a topic to every seat in the Agent Room and bring the chair's summary back here.
disable-model-invocation: true
---

# /agent-rooms \<topic\>

The room decides and the user closes the discussion. Your job is to carry the outcome back with its **provenance** intact, meaning the label that says whether the outcome is an agreed decision or something less.

`ROOM` is the install folder, and `AGENT_ROOM_WORKDIR` is the folder the agents work in. Both are set in the commands below.

## 1. Take the topic

The topic is everything after `/agent-rooms`. If nothing follows it, ask for the topic once.

Done when: you hold a non-empty topic.

## 2. Open the room in the background

The user may take an hour to close, which is longer than one shell call can wait. So start the command detached:

```bash
ROOM="__ROOM__"
OUT="$ROOM/logs/last-boardroom-outcome.txt"
mkdir -p "$ROOM/logs"
rm -f "$OUT"
AGENT_ROOM_WORKDIR="__WORKDIR__" nohup python3 "$ROOM/agent_rooms.py" "<topic>" --outcome-file "$OUT" \
  >"$ROOM/logs/last-boardroom-run.log" 2>&1 &
```

The command:

1. starts the server, or attaches to one that is already running
2. opens a fresh chat named after the topic
3. invites every seat
4. waits for the user to close
5. writes the outcome to `$OUT`

Tell the user once that the room is open and this session is waiting on it.

Done when: the process is running and the user knows the room is open.

## 3. Check back with short reads

Check whenever the user says the room is closed, and between turns:

```bash
ROOM="__ROOM__"
cat "$ROOM/logs/last-boardroom-outcome.txt" 2>/dev/null \
  || tail -n 5 "$ROOM/logs/last-boardroom-run.log"
```

If the run log shows `Error:`, the server is down or the wait timed out. Report that error in plain words and finish.

Done when: you hold the full outcome text or a clear error.

## 4. Report with provenance

Restate the outcome here with its `summary_source` label translated:

| `summary_source` | Tell the user |
|---|---|
| `chair` | This is the agreed outcome, the chair's summary. |
| `unsummarised-chair-remark` | This is a chair aside, not an agreed decision. |
| `placeholder` | There is no agreed outcome. The room closed before a summary. |
| anything else | The provenance is unknown. Treat it as not agreed until checked. |

Done when: the user can read the outcome and its provenance here without opening the room.
