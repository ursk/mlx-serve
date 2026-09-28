#!/bin/bash
# A hybrid follow-up request restores past the previous reply it resends instead of re-prefilling it.
#
# Turn A asks for a tool call; turn B sends A back with the tool result. B token-matches
# A's cache entry (prompt ++ reply) to the reply's last token, so B must report
# cached_tokens == A.prompt_tokens + A.completion_tokens. Before the end-of-generation
# checkpoint a hybrid restore stopped at the last prefill checkpoint, ~30 tokens before
# A's prompt end. Two shapes: A decoding alone (serial pipeline, which has run past the
# reply's end when it stops, so the state comes from `Generator.held_ssm`) and A decoding
# beside a long request (batched tick). The log must show one decode-end checkpoint per A.
# A checkpoint that ships an MTP
# head runs both shapes again under --mtp, whose rounds stop the trunk before an accepted EOS,
# plus a third pair whose B drafts: its head must resume from A's history at the reply's end.
#
# Then each pass's serial B's greedy text is compared with a cold server (`--prefix-cache-entries 0`).
# A restore is not bit-exact with a computed prefix, so a mismatch passes only when the
# first differing token is a near-tie on the cold run (top-2 gap <= 0.15 nats). The batched
# A's state comes from a batched tick, which differs from a solo prefill with or without
# this checkpoint, so that case is not compared with cold.
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
    # --no-mtp outranks a later --mtp (the head loads but requests never draft), so a pass asking
    # for --mtp must not get it.
    local mtp=--no-mtp a
    for a in "$@"; do [ "$a" = --mtp ] && mtp=; done
    "$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 --prefix-cache-disk off \
        --no-pld $mtp "$@" ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -40 "$LOGFILE"; exit 1
}

# One python helper for every request shape: `a` (turn A, optionally beside a long
# filler), `b` (turn B from A's saved reply, greedy with top-2 logprobs).
cat > "$WORK/turn.py" <<'PY'
import json, sys, threading, time, urllib.request
base, mode, case, work = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a fact.",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
TOPIC = {"serial": "the boiling point of water at sea level", "batched": "the speed of light in vacuum",
         "head": "the melting point of iron"}

def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def first_turn():
    return [{"role": "user", "content": f"Use the lookup tool to find {TOPIC[case]}, then answer in one sentence."}]

if mode == "a":
    filler = None
    if case == "batched":
        # Long under either decoder: MTP's greedy essay ended a third of the way into plain's.
        fbody = {"messages": [{"role": "user", "content": "Count from one to three thousand in words, one number per line."}],
                 "max_tokens": 3000, "temperature": 0}
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
    body = {"messages": msgs, "tools": TOOLS, "max_tokens": 256, "temperature": 0}
    if case == "serial":  # the warm-vs-cold check reads these; logprobs turn speculative decoding off
        body.update(logprobs=True, top_logprobs=2)
    r = post("/v1/chat/completions", body)
    u = r["usage"]
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
    print(json.dumps({"entry_end": a["usage"]["prompt_tokens"] + a["usage"]["completion_tokens"],
                      "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                      "filler_alive": a["filler_alive"], "text": r["choices"][0]["message"].get("content") or "",
                      "lp": [[t["token"], [[x["token"], x["logprob"]] for x in t.get("top_logprobs") or []]] for t in lp]}))
PY

fail=0
# One warm pass per decode path: plain, and MTP when the checkpoint ships a head.
PASSES="plain"
grep -q '"language_model.mtp.fc_hidden.weight"' "$MODEL/model.safetensors.index.json" 2>/dev/null && PASSES="plain mtp"
for pass in $PASSES; do
    W="$WORK/$pass"; mkdir -p "$W"
    # MTP adds `head`, a second serial pair whose B asks for no logprobs, so B drafts with the head.
    CASES="serial batched"
    if [ "$pass" = mtp ]; then start_server --prefix-cache-entries 8 --prefix-cache-mem 4096MB --max-concurrent 2 --mtp; CASES="serial batched head"
    else start_server --prefix-cache-entries 8 --prefix-cache-mem 4096MB --max-concurrent 2; fi
    for case in $CASES; do
        python3 "$WORK/turn.py" "$BASE" a "$case" "$W" || { echo -e "${RED}FAIL${NC} $pass $case: turn A"; fail=1; continue; }
        python3 "$WORK/turn.py" "$BASE" b "$case" "$W" > "$W/$case.warm.json" || { echo -e "${RED}FAIL${NC} $pass $case: turn B"; fail=1; continue; }
        if python3 -c "
import json,sys; o=json.load(open(sys.argv[1]))
print('  %s: B cached %d, A entry ends at %d, filler alive at A end: %s' % (sys.argv[2], o['cached'], o['entry_end'], o['filler_alive']))
raise SystemExit(0 if o['cached'] == o['entry_end'] else 1)" "$W/$case.warm.json" "$pass $case"; then
            echo -e "${GREEN}PASS${NC} $pass $case: turn B restored at the end of A's reply"
        else
            echo -e "${RED}FAIL${NC} $pass $case: turn B re-prefilled part of A's reply"; fail=1
        fi
    done
    if [ "$pass" = mtp ]; then  # the batched and head Bs draft; the serial B's logprobs turn drafting off
        if grep -q "MTP head restored" "$LOGFILE" && ! grep -q "not adopted" "$LOGFILE"; then
            echo -e "${GREEN}PASS${NC} mtp: turn B's draft head resumed from A's history"
        else
            grep -E "MTP head restored|not adopted" "$LOGFILE" | sed 's/^/  /'
            echo -e "${RED}FAIL${NC} mtp: turn B's draft head started without A's history"; fail=1
        fi
    fi
    n_cp=$(grep -c "decode-end checkpoint at" "$LOGFILE" || true)
    n_a=$(echo $CASES | wc -w)
    if [ "$n_cp" -ge "$n_a" ]; then
        echo -e "${GREEN}PASS${NC} $pass: a decode-end checkpoint per reply ($n_cp for $n_a turn As)"
    else
        echo -e "${RED}FAIL${NC} $pass: $n_cp decode-end checkpoints for $n_a turn As"; fail=1
    fi
    stop_server
done

start_server --prefix-cache-entries 0
for pass in $PASSES; do
    W="$WORK/$pass"
    [ -f "$W/serial.warm.json" ] || continue
    python3 "$WORK/turn.py" "$BASE" b serial "$W" > "$W/serial.cold.json" || { echo -e "${RED}FAIL${NC} $pass: cold turn B"; fail=1; continue; }
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
raise SystemExit(0 if gap <= 0.15 else 1)" "$W/serial.warm.json" "$W/serial.cold.json" "$pass serial"; then
        echo -e "${GREEN}PASS${NC} $pass serial: warm turn B decodes like a cold prefill"
    else
        echo -e "${RED}FAIL${NC} $pass serial: warm turn B diverges from cold at a confident token"; fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} test_prefix_cache_gen_end"
    exit 0
fi
exit 1
