# Agent Room

Agent Room is a group chat in your browser where Claude Code, Grok and Kimi work together while you watch.

Everyone reads one feed.
`@name` sends a message to one agent, and the agents tag each other the same way.
A Mission is a goal plus a done-check. The room loops plan (claude), build (grok) and review (kimi) until the check passes, and the Stop button works the whole time.

It runs locally on the Python standard library and drives the agent CLIs you already pay for.

## Install

Paste this into Claude Code or Codex:

```text
Install Agent Room for me by following https://github.com/kuhnhomeuk-cell/agent-room/blob/main/INSTALL.md
```

[INSTALL.md](INSTALL.md) covers what you need (Python 3.11+ and Claude Code, with Grok and Kimi optional), fits the seats to the CLIs you have, and proves the room runs.
The same steps work by hand.

## Day to day

```bash
AGENT_ROOM_WORKDIR=~/my-project ~/agent-room/start.sh
```

Then open http://127.0.0.1:8787.

- `AGENT_ROOM_WORKDIR` is the folder the agents work in.
- You edit seats in `agents.json` or live from the sidebar. The file's `_help` field explains each field.
- Your chats are saved in `rooms/` and never committed.
- Run the tests with `python3 -m unittest discover`.
