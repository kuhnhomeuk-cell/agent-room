# Install Agent Room

You are the coding agent setting Agent Room up for the person in front of you.
Work the steps in order.
A step is finished only when its **Done when** line holds on this machine. You prove it with a command you ran, not with an assumption.
Shell variables do not survive between your commands, so open every command block with the two assignments:

```bash
ROOM="$HOME/agent-room"      # replace with the clone folder from step 1
WORK="$HOME/my-project"      # replace with the working folder from step 4
```

## 1. Clone

Ask where they want it. The default is `~/agent-room`, and `ROOM` holds the answer.

```bash
git clone https://github.com/kuhnhomeuk-cell/agent-room.git "$ROOM"
```

Done when: `$ROOM/server.py` exists.

## 2. Inventory the machine

Agent Room is Python standard library only. There is no package install.

```bash
python3 --version
command -v curl claude grok kimi
claude auth status
```

- Python must be 3.11 or newer, because 3.10 cannot read Kimi's model list.
- `claude` is required. It chairs missions and writes the minutes. If it is missing or signed out, the person installs Claude Code and runs `/login` inside `claude`.
- `grok` and `kimi` are optional. Record which ones are present. Signing in to them happens later, from the room's sidebar.

Done when: Python is 3.11 or newer, `curl` is present, `claude auth status` reports signed in, and you hold the list of engines that are on PATH.

## 3. Fit the seats to that list

Seats live in `$ROOM/agents.json`, and its `_help` field defines every seat field.
Every seat whose `engine` is missing from the list in step 2 gets `"engine": "claude"`, `"model": null` and `"effort": null`. Its `name`, `color` and `role` stay as they are.

Done when: every `engine` value in `agents.json` is on the step 2 list, and `python3 -m json.tool "$ROOM/agents.json"` parses.

## 4. Pick the working folder

The agents read and write files in one folder, which is `AGENT_ROOM_WORKDIR`.
Ask the person which project folder that is, and hold it in `WORK`. Agents on missions edit files there, so it should be a folder they're happy to have changed.
If the variable is unset, the room uses the folder `start.sh` was run from.

Done when: the person has named a folder, and it exists.

## 5. Prove it runs

```bash
python3 -m unittest discover -s "$ROOM"
AGENT_ROOM_WORKDIR="$WORK" "$ROOM/start.sh"
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"text":"@claude run pwd and reply with only the path"}' \
  http://127.0.0.1:8787/api/send
curl -s http://127.0.0.1:8787/api/messages
```

The reply takes up to a minute. Re-read `/api/messages` until a message `"from": "claude"` appears.
If `start.sh` fails, it prints the end of `$ROOM/logs/server.log`, which is where the cause is.

Done when: the tests end `OK`, `start.sh` prints `Agent Room is running at http://127.0.0.1:8787`, and claude's reply is the working folder from step 4.

## 6. Offer the `/agent-rooms` skill (Claude Code only)

The skill puts one topic to every seat from inside a Claude Code session and brings the chair's summary back.
Install it only on the person's yes.

```bash
mkdir -p ~/.claude/skills/agent-rooms
sed -e "s|__ROOM__|$ROOM|g" -e "s|__WORKDIR__|$WORK|g" \
  "$ROOM/skills/agent-rooms/SKILL.md" > ~/.claude/skills/agent-rooms/SKILL.md
```

Done when: the person said no, or the installed file has no `__ROOM__` or `__WORKDIR__` left, and every `ROOM=` and `AGENT_ROOM_WORKDIR=` path in it is a folder that exists.

## 7. Hand over

Give the person these four things in plain words:

1. The start command: `AGENT_ROOM_WORKDIR="<WORK>" <ROOM>/start.sh`, then http://127.0.0.1:8787. If the room is already running, it keeps its current folder until `server.py` is stopped.
2. How to talk to the room: `@name` reaches one seat, an untagged message goes to `@claude`, and a Mission (a goal plus a done-check) loops plan, build and review until the check passes. Stop ends it at any point.
3. Cost: every agent turn runs on their own CLI subscriptions.
4. Privacy: chats are saved in `<ROOM>/rooms/` and stay on this machine.

Done when: the person has all four and has seen the room open in their browser.
