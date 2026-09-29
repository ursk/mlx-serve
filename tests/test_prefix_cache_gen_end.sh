#!/bin/bash
# A hybrid follow-up request restores past the previous reply it resends instead of re-prefilling it.
#
# Turn A asks for a tool call; turn B sends A back with the tool result. B token-matches
# A's cache entry (prompt ++ reply) to the reply's last token, so B must report
# cached_tokens == A.prompt_tokens + A.completion_tokens. Before the end-of-generation
# checkpoint a hybrid restore stopped at the last prefill checkpoint, ~30 tokens before
# A's prompt end. Two shapes: A decoding alone (serial pipeline, which has run past the
# reply's end when it stops) and A decoding beside a long request (batched tick).
# `/metrics.json` must count one checkpoint per A and no skips.
#
# Then the serial B's greedy text is compared with a cold server (`--prefix-cache-entries 0`).
# A restore is not bit-exact with a computed prefix, so a mismatch passes only when the
# first differing token is a near-tie on the cold run (top-2 gap <= 0.15 nats). The batched
# A's state comes from a batched tick, which differs from a solo prefill with or without
# this checkpoint, so that case is not compared with cold.
# MLX_SERVE_GEN_END_CHECKPOINT=0 turns the checkpoint off: the cached-token checks go red.
#
# Warm and cold run sequentially on ONE port.
# Usage: ./tests/test_prefix_cache_gen_end.sh [/path/to/qwen4_exp or qwen3_5 hybrid] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}"
PORT="${2:-11491}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_cache_gen_end: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

SERVER_PID=""
LOGFILE=$(mktemp)
WORK=$(mktemp -d)

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    for _ in $(seq 1 60); do lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
}
cleanup() { stop_server; rm -rf "$LOGFILE" "$WORK"; }
trap cleanup EXIT INT TERM

start_server() {
    echo "  starting server $*..."
    "$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 --prefix-cache-disk off \
        --no-pld --no-mtp --metrics "$@" ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -40 "$LOGFILE"; exit 1
}

# One python helper for every request shape: `a` (turn A, optionally beside a long
# filler), `b` (turn B from A's saved reply, greedy with top-2 logprobs), `metrics`.
cat > "$WORK/turn.py" <<'PY'
import json, sys, threading, time, urllib.request
base, mode, case, work = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a fact.",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
TOPIC = {"serial": "the boiling point of water at sea level", "batched": "the speed of light in vacuum"}

def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def first_turn():
    return [{"role": "user", "content": f"Use the lookup tool to find {TOPIC[case]}, then answer in one sentence."}]

if mode == "metrics":
    m = json.load(urllib.request.urlopen(base + "/metrics.json", timeout=30))
    c = m.get("counters", m)
    print(json.dumps({k: c.get(k) or 0 for k in ("prefix_cache_gen_end_checkpoints_total",
                                                 "prefix_cache_gen_end_skipped_total")}))
elif mode == "a":
    filler = None
    if case == "batched":
        fbody = {"messages": [{"role": "user", "content": "Write a 1500-word essay on the history of the telescope."}],
                 "max_tokens": 1500, "temperature": 0}
        filler = threading.Thread(target=post, args=("/v1/chat/completions", fbody))
        filler.start()
        time.sleep(8)  # the filler is decoding before A arrives
    r = post("/v1/chat/completions", {"messages": first_turn(), "tools": TOOLS, "max_tokens": 1024, "temperature": 0})
    alive = filler.is_alive() if filler else None
    if filler:
        filler.join()
    msg = r["choices"][0]["message"]
    json.dump({"msg": msg, "usage": r["usage"], "filler_alive": alive}, open(f"{work}/{case}.a.json", "w"))
    assert msg.get("tool_calls"), f"turn A made no tool call: {msg}"
else:
    a = json.load(open(f"{work}/{case}.a.json"))
    m = a["msg"]
    asst = {"role": "assistant", "content": m.get("content") or None, "tool_calls": m["tool_calls"]}
    if m.get("reasoning_content"):
        asst["reasoning_content"] = m["reasoning_content"]
    msgs = first_turn() + [asst, {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"],
                                  "content": "Found it; answer from your own knowledge."}]
    r = post("/v1/chat/completions", {"messages": msgs, "tools": TOOLS, "max_tokens": 256, "temperature": 0,
                                      "logprobs": True, "top_logprobs": 2})
    u = r["usage"]
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
    print(json.dumps({"entry_end": a["usage"]["prompt_tokens"] + a["usage"]["completion_tokens"],
                      "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                      "filler_alive": a["filler_alive"], "text": r["choices"][0]["message"].get("content") or "",
                      "lp": [[t["token"], [[x["token"], x["logprob"]] for x in t.get("top_logprobs") or []]] for t in lp]}))
PY

fail=0
start_server --prefix-cache-entries 8 --prefix-cache-mem 4096MB --max-concurrent 2
python3 "$WORK/turn.py" "$BASE" metrics - "$WORK" > "$WORK/m0.json"
for case in serial batched; do
    python3 "$WORK/turn.py" "$BASE" a "$case" "$WORK" || { echo -e "${RED}FAIL${NC} $case: turn A"; fail=1; continue; }
    python3 "$WORK/turn.py" "$BASE" b "$case" "$WORK" > "$WORK/$case.warm.json" || { echo -e "${RED}FAIL${NC} $case: turn B"; fail=1; continue; }
    if python3 -c "
import json,sys; o=json.load(open(sys.argv[1]))
print('  %s: B cached %d, A entry ends at %d, filler alive at A end: %s' % (sys.argv[2], o['cached'], o['entry_end'], o['filler_alive']))
raise SystemExit(0 if o['cached'] == o['entry_end'] else 1)" "$WORK/$case.warm.json" "$case"; then
        echo -e "${GREEN}PASS${NC} $case: turn B restored at the end of A's reply"
    else
        echo -e "${RED}FAIL${NC} $case: turn B re-prefilled part of A's reply"; fail=1
    fi
done
if python3 "$WORK/turn.py" "$BASE" metrics - "$WORK" | python3 -c "
import json,sys; m1=json.load(sys.stdin); m0=json.load(open(sys.argv[1]))
d={k: m1[k]-m0[k] for k in m1}; print('  metrics delta', d)
raise SystemExit(0 if d['prefix_cache_gen_end_checkpoints_total'] >= 2 and d['prefix_cache_gen_end_skipped_total'] == 0 else 1)" "$WORK/m0.json"; then
    echo -e "${GREEN}PASS${NC} one end-of-generation checkpoint per reply, none skipped"
else
    echo -e "${RED}FAIL${NC} end-of-generation checkpoint counters"; fail=1
fi
stop_server

start_server --prefix-cache-entries 0
for case in serial; do
    [ -f "$WORK/$case.warm.json" ] || continue
    python3 "$WORK/turn.py" "$BASE" b "$case" "$WORK" > "$WORK/$case.cold.json" || { echo -e "${RED}FAIL${NC} $case: cold turn B"; fail=1; continue; }
    if python3 -c "
import json,sys
w=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
wt=[t[0] for t in w['lp']]; ct=[t[0] for t in c['lp']]
if wt == ct:
    print('  %s: warm == cold (%d tokens)' % (sys.argv[3], len(wt))); raise SystemExit(0)
i=next((k for k,(x,y) in enumerate(zip(wt,ct)) if x!=y), min(len(wt),len(ct)))
if i >= len(ct):
    print('  %s: cold ended at %d, warm continued' % (sys.argv[3], i)); raise SystemExit(1)
top=c['lp'][i][1]
gap=top[0][1]-top[1][1] if len(top) >= 2 else float('inf')
print('  %s: first difference at token %d of %d, cold top-2 gap %.3f nats' % (sys.argv[3], i, len(ct), gap))
raise SystemExit(0 if gap <= 0.15 else 1)" "$WORK/$case.warm.json" "$WORK/$case.cold.json" "$case"; then
        echo -e "${GREEN}PASS${NC} $case: warm turn B decodes like a cold prefill"
    else
        echo -e "${RED}FAIL${NC} $case: warm turn B diverges from cold at a confident token"; fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} test_prefix_cache_gen_end"
    exit 0
fi
exit 1
