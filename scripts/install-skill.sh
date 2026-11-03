#!/bin/sh
# Install the voice-normal Agent Skill for Claude Code and Codex on macOS / Linux.
#
#   scripts/install-skill.sh            # symlink into ~/.claude/skills and ~/.agents/skills
#   scripts/install-skill.sh --copy     # copy instead of linking
#   scripts/install-skill.sh --uninstall
set -eu

mode=link
action=install
for arg in "$@"; do
    case "$arg" in
        --copy) mode=copy ;;
        --uninstall) action=uninstall ;;
        *) echo "usage: $0 [--copy] [--uninstall]" >&2; exit 2 ;;
    esac
done

repo=$(cd "$(dirname "$0")/.." && pwd)
source="$repo/skills/voice-normal"
home=${VOICE_NORMAL_SKILL_HOME:-$HOME}
status=0

[ "$action" = install ] && printf '%s\n' "$repo" > "$source/repo-root.txt"

for target in "$home/.claude/skills/voice-normal" "$home/.agents/skills/voice-normal"; do
    if [ -L "$target" ]; then
        if [ "$(cd "$target" && pwd -P)" != "$(cd "$source" && pwd -P)" ]; then
            echo "[!!] $target links somewhere else; remove it yourself." >&2; status=1; continue
        fi
        rm "$target"
    elif [ -e "$target" ]; then
        if [ ! -f "$target/repo-root.txt" ]; then
            echo "[!!] $target exists and is not a voice-normal install; remove it yourself." >&2; status=1; continue
        fi
        rm -r "$target"
    elif [ "$action" = uninstall ]; then
        echo "[--] not installed: $target"; continue
    fi
    if [ "$action" = uninstall ]; then
        echo "[OK] removed $target"; continue
    fi
    mkdir -p "$(dirname "$target")"
    if [ "$mode" = link ]; then
        ln -s "$source" "$target"
    else
        cp -R "$source" "$target"
        printf '%s\n' "$repo" > "$target/repo-root.txt"
    fi
    echo "[OK] $target ($mode -> $source)"
done

[ "$action" = uninstall ] && rm -f "$source/repo-root.txt"
exit $status
