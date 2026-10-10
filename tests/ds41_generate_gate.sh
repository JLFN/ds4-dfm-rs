#!/bin/sh
# ds41 greedy-generate gate runner (unit E, E0): one positive run + two
# REQUIRED failing negative controls — a gate that only ever passes cannot
# tell correct from blind.
#
#   positive            PASS with full row-reference coverage (emitted N/N,
#                       0 mismatches, both engram layers);
#   DS41_EMIT_OFFSET=3  the E0 class (the feed's declared position drifts):
#                       the row reference must MISMATCH and the port must
#                       refuse the step ("engram feed pos0 ... != ...");
#   DS41_ROW_SHIFT=1    the declared position stays right but the rows hashed
#                       are for the wrong position: the row reference must
#                       MISMATCH while the port's position assertion stays
#                       quiet (the two defenses are independent).
#
# usage: ds41_generate_gate.sh <harness> <gguf> <ids> <englog> <n> <engram_dir> <rows_ref>
set -u

H="$1" MODEL="$2" IDS="$3" ENGLOG="$4" N="$5" ENGRAM="$6" REF="$7"
LOG=/tmp/ds41_gen_gate

fail() { echo "ds41 generate gate: FAIL ($1)"; exit 1; }

[ -n "$REF" ] || fail "ROWS_REF is required (the emitted positions must be gated)"
ARGS="--engram-dir $ENGRAM --rows-ref $REF"

"$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.positive.log" 2>&1 \
  || fail "positive run exited nonzero (see $LOG.positive.log)"
grep -q "DS41 generate gate: PASS" "$LOG.positive.log" || fail "positive run did not PASS"
[ "$(grep -c "emitted $N/$N checked, 0 mismatches" "$LOG.positive.log")" -eq 2 ] \
  || fail "positive run lacks full emitted row-reference coverage ($N/$N x2 layers, 0 mismatches)"

DS41_EMIT_OFFSET=3 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.emit_offset.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control DS41_EMIT_OFFSET=3 PASSED (blind gate)"
grep -q "MISMATCH" "$LOG.emit_offset.log" || fail "DS41_EMIT_OFFSET=3: the row reference did not fire"
grep -q "engram feed pos0" "$LOG.emit_offset.log" || fail "DS41_EMIT_OFFSET=3: the port did not refuse the drifted feed"

DS41_ROW_SHIFT=1 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.row_shift.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control DS41_ROW_SHIFT=1 PASSED (blind row reference)"
grep -q "MISMATCH" "$LOG.row_shift.log" || fail "DS41_ROW_SHIFT=1: the row reference did not fire"
grep -q "engram feed pos0" "$LOG.row_shift.log" && fail "DS41_ROW_SHIFT=1: the port refused too — the two controls are not independent"

echo "ds41 generate gate: positive PASS ($N/$N emitted, 0 mismatches), negative controls FAIL as required"
