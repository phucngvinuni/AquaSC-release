"""Torch-side client for the Sionna digital-pipeline bridge (Wave 5 CHANNEL-cont).

Spawns a persistent subprocess running digital_bridge.py under the sionna venv
and speaks JSON-lines over stdin/stdout. Keeps the process alive across the
whole eval sweep so the ~5 s TensorFlow import cost is paid once, not once per
image.
"""
from __future__ import annotations

import json
import os
import subprocess

import torch

_SERVER = None
_BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digital_bridge.py")
_OVERRIDE_PYTHON = None


def set_bridge_python(path):
    global _OVERRIDE_PYTHON
    _OVERRIDE_PYTHON = path


def _bridge_python():
    return (_OVERRIDE_PYTHON or os.environ.get("DIGITAL_BRIDGE_PYTHON")
            or "__SIONNA_PY__")


def _ensure_server():
    global _SERVER
    if _SERVER is not None and _SERVER.poll() is None:
        return _SERVER
    py = _bridge_python()
    if not os.path.isfile(_BRIDGE_SCRIPT):
        raise RuntimeError(f"digital bridge script not found: {_BRIDGE_SCRIPT}")
    if not os.path.isfile(py):
        raise RuntimeError(f"sionna venv python not found: {py}")
    log_f = open("__AQUASC_DATA__/logs/digital_bridge_stderr.log", "ab")
    _SERVER = subprocess.Popen(
        [py, _BRIDGE_SCRIPT, "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_f,
        text=True, bufsize=1,
    )
    resp = _request({"op": "ping"})
    if not resp.get("ok"):
        raise RuntimeError(f"digital bridge handshake failed: {resp}")
    return _SERVER


def _request(obj):
    srv = _ensure_server()
    srv.stdin.write(json.dumps(obj) + "\n")
    srv.stdin.flush()
    line = srv.stdout.readline()
    if not line:
        raise RuntimeError("digital bridge process died (empty reply); see logs/digital_bridge_stderr.log")
    return json.loads(line)


def shutdown():
    global _SERVER
    if _SERVER is not None and _SERVER.poll() is None:
        try:
            _request({"op": "shutdown"})
        except Exception:
            pass
        try:
            _SERVER.stdin.close()
            _SERVER.wait(timeout=10)
        except Exception:
            pass
    _SERVER = None


def transmit_indices_digital(token_values, high_mask, b_high, b_low, snr_db,
                             csi_mode="perfect", seed=42,
                             rvq_stage_uep=False, return_stage_status=False,
                             bits_per_symbol=4, payload_rate=0.5, joint_frame=False):
    """Transmit B images' token indices through the real digital chain.

    Args:
        token_values: [B, n_tokens] long tensor on any device (packed RVQ index for
            high tokens, flat index for low tokens; n_tokens = 784 for patch-8,
            196 for patch-16).
        high_mask: [B, n_tokens] bool tensor.
        b_high, b_low: per-token bit widths.
        snr_db, csi_mode, seed.

    Returns ``(recovered, stats)`` normally. With ``rvq_stage_uep`` and
    ``return_stage_status``, returns ``(recovered, stats, stage_values,
    stage_erased, low_values, low_erased)`` so a partially erased RVQ token can
    be reconstructed from the stages that survived.
    Recovered values are the per-token indices with **-1 marking erased tokens**
    (vector-domain N4 concealment happens on the torch side in model.py).
    stats = {blocks, failed, bler, ber, bit_accuracy, erased_tokens,
             erased_token_fraction, iterations_mean, n_images}.
    """
    device = token_values.device
    tv = token_values.detach().cpu().numpy()
    hm = high_mask.detach().cpu().numpy()
    B = tv.shape[0]
    images = []
    for i in range(B):
        images.append({
            "tokens": tv[i].astype(int).tolist(),
            "high_mask": hm[i].astype(int).tolist(),
            "b_high": int(b_high), "b_low": int(b_low),
        })
    resp = _request({
        "op": "transmit", "snr_db": float(snr_db), "csi_mode": csi_mode,
        "seed": int(seed), "images": images, "rvq_stage_uep": bool(rvq_stage_uep),
        "bits_per_symbol": int(bits_per_symbol), "payload_rate": float(payload_rate),
        "joint_frame": bool(joint_frame),
    })
    if not resp.get("ok"):
        raise RuntimeError(f"digital bridge transmit failed: {resp}")
    recovered = torch.tensor(
        [r["recovered"] for r in resp["images"]], dtype=torch.long, device=device)
    if rvq_stage_uep and return_stage_status:
        stage_values = torch.tensor(
            [r["stage_values"] for r in resp["images"]], dtype=torch.long,
            device=device)
        stage_erased = torch.tensor(
            [r["stage_erased"] for r in resp["images"]], dtype=torch.bool,
            device=device)
        low_values = torch.tensor(
            [r["low_values"] for r in resp["images"]], dtype=torch.long,
            device=device)
        low_erased = torch.tensor(
            [r["low_erased"] for r in resp["images"]], dtype=torch.bool,
            device=device)
        return recovered, resp["stats"], stage_values, stage_erased, low_values, low_erased
    return recovered, resp["stats"]
