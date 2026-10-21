#!/bin/bash
# Recovery evaluation for one trained ladder arm, on the identical protocol the
# frozen 5,957,984-parameter baseline was measured with: 100 test datasets x 20
# subjects x 256 draws x 50 DPM steps, corrector-free, KDE-global-then-local MAP.
# The three inference methods are the ones under study:
#   jacobian_newton  DPM2 + gauss_jacobian (pilot-free), churn 2, refresh 20
#   hier_churn2      DPM2 + gauss_hierarchical, churn 2
#   fnpse            Langevin + F-NPSE (no churn; it never enters _dpm_sampler)
#   jacobian_fd_newton  DPM2 + gauss_jacobian_FD, churn 2, refresh $REFRESH (default 20),
#                       or $LOW_REFRESH below lambda 0.3 when set
#
# Usage: eval_lane.sh <arm-name> <method> -- <model selection flags...>
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python

ARM=$1; METHOD=$2; shift 3   # third argument is the literal --
LOGS=$PP/ladder/logs
mkdir -p "$LOGS"

COMMON=(--preset small --datasets 100 --subjects 20 --draws 256 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

case "$METHOD" in
  jacobian_newton)
    CHURN=(--churn-eta 2 --jacobian-refresh 20
           --inference-method dpm2_gauss_jacobian_newton) ;;
  jacobian_fd_newton)
    # FD Jacobian at a uniform refresh period (REFRESH, default 20).
    # LOW_REFRESH, if set, is the period below lambda 0.3 (REFRESH above it).
    CHURN=(--churn-eta 2 --jacobian-refresh "${REFRESH:-20}"
           ${LOW_REFRESH:+--jacobian-refresh-low-noise "$LOW_REFRESH"}
           --inference-method dpm2_gauss_jacobian_FD_newton)
    METHOD="jacobian_fd_newton_refresh${REFRESH:-20}${LOW_REFRESH:+_low$LOW_REFRESH}" ;;
  hier_churn2)
    CHURN=(--churn-eta 2 --inference-method dpm2_gauss_hierarchical) ;;
  fnpse)
    CHURN=(--churn-eta 0 --inference-method langevin_fnpse) ;;
  *) echo "unknown method $METHOD" >&2; exit 2 ;;
esac

echo "=== [$(date -Is)] starting $ARM/$METHOD" >&2
"$PY" "$PP/run_churn_comparison.py" \
    --scratch-base "$PP/ladder/$ARM" "${CHURN[@]}" "$@" "${COMMON[@]}" \
    >"$LOGS/eval-$ARM-$METHOD.log" 2>&1
echo "=== [$(date -Is)] $ARM/$METHOD exit $?" >&2
