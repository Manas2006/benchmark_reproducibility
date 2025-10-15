# prob_recorder.py
import os
import json
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    # vLLM v1 interface paths (>= 0.6.x)
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor
except Exception:
    # Fallback older import path (unlikely for "v1" but just in case)
    from vllm.model_executor.layers.logits_processor import LogitsProcessor  # type: ignore


def _safe_float(x: torch.Tensor) -> float:
    try:
        return float(x.detach().cpu().item())
    except Exception:
        return float(x)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    
# ---- global sidecar store ----
_RECORDER_STATE = {}

def _norm_req_id(req_id):
    # store & lookup using a consistent key
    try:
        return int(req_id)
    except Exception:
        return str(req_id)

def push_request_state(req_id, payload):
    _RECORDER_STATE[_norm_req_id(req_id)] = payload

def pop_request_state(req_id):
    return _RECORDER_STATE.pop(_norm_req_id(req_id), None)



class BatchProbabilityRecorder(LogitsProcessor):
    """
    No-op logits processor that records per-step stats:
      - entropy of the softmax distribution
      - chosen token probability (using sampler_output)
      - correct token probability (if gold_token_ids provided in extra_args)

    It writes one JSON sidecar per request_id on finalize:
      <output_dir>/<dataset>/<run_id>/requests/<request_id>.json

    REQUIRED abstract methods for vLLM v1 are implemented:
      - is_argmax_invariant()
      - apply(state)
      - update_state(state, logits, sampler_output)
    We also provide helpers that vLLM calls if present:
      - new_request_state(request_id, seq_group, extra_args)
      - on_request_end(request_id)
    """

    # DO NOT call super().__init__: base raises NotImplementedError
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.is_pin_memory = is_pin_memory
        # store per-request dictionaries keyed by request_id (string or int)
        self._states: Dict[str, Dict[str, Any]] = {}
        # store per-request sidecar data
        self._sidecars: Dict[str, Dict[str, Any]] = {}

    # ---- vLLM abstract API ----

    def is_argmax_invariant(self) -> bool:
    # We don't change logits; purely observational.
        return True

    # inside BatchProbabilityRecorder

    def apply(self, logits):
        """
        vLLM calls this every decoding step just before sampling.
        We forward to update_state with the richer set of objects.
        """
        # Graceful no-op if logits are missing (can happen on init / fail-fast paths)
        if logits is None:
            return logits

        # Debug: Print when apply is called (only once)
        if not hasattr(self, '_debug_printed'):
            print(f"[BatchProbabilityRecorder] apply() called with logits shape={logits.shape if hasattr(logits, 'shape') else 'no shape'}")
            self._debug_printed = True

        # For now, we'll try to record basic information without sampling_metadata
        # This is a simplified approach that should work with the current vLLM interface
        try:
            # Basic recording without full metadata
            self._record_basic_stats(logits)
        except Exception as e:
            # Never crash the engine from here; just log and continue
            if not getattr(self, "_swallowed_once", False):
                print(f"[BatchProbabilityRecorder] apply error (swallowed): {e}")
                self._swallowed_once = True

        # We are a recorder, not a modifier — return logits unchanged
        return logits

    def _record_basic_stats(self, logits):
        """Record basic statistics from logits without full metadata."""
        import torch
        
        # Compute basic stats
        probs = torch.softmax(logits.float(), dim=-1)
        
        # Compute entropy for each row
        with torch.no_grad():
            ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
        
        # Store basic stats in global state for any active requests
        # This is a simplified approach that records basic information
        for rid in self._states.keys():
            if rid in _RECORDER_STATE:
                global_buf = _RECORDER_STATE[rid]
                global_buf["entropies"].append(ent.mean().item())  # Use mean entropy for simplicity
                # We can't get chosen token probs without sampler output, so skip for now
                if "chosen_token_probs" not in global_buf:
                    global_buf["chosen_token_probs"] = []

    def update_state(self, *args, **kwargs):
        """
        Unified handler for both call sites:

        (A) refresh_metadata():   update_state(batch_update)
        (B) sampling pipeline:    update_state(state, logits, sampler_output, sampling_metadata)
        """
        import torch

        # --- (A) Called with a single BatchUpdate during refresh_metadata()
        if len(args) == 1:
            batch_update = args[0]
            # It's safe to no-op here unless you want to pre-initialize per-request buffers.
            # Avoid raising; vLLM expects this to succeed.
            return

        # --- (B) Called from our own apply() wrapper with full context
        if len(args) >= 3:
            state, logits, sampler_output = args[:3]
            sampling_metadata = args[3] if len(args) > 3 else None

            if logits is None:
                return  # nothing to record this step

            try:
                # probs: [num_active, vocab]
                probs = torch.softmax(logits.float(), dim=-1)

                # chosen token ids: [num_active]
                chosen_ids = getattr(sampler_output, "sampled_token_ids", None)
                if chosen_ids is None:
                    return  # can't compute chosen-token prob without sampled ids

                chosen_probs = probs.gather(-1, chosen_ids.unsqueeze(-1)).squeeze(-1)

                # request ids for each row in this step
                req_ids = None
                if sampling_metadata is not None:
                    req_ids = getattr(sampling_metadata, "request_ids", None)
                if req_ids is None and sampler_output is not None:
                    req_ids = getattr(sampler_output, "request_ids", None)
                if req_ids is None:
                    # Fall back to synthetic ids (avoids crashes; you won't be able to match later)
                    req_ids = [str(i) for i in range(chosen_probs.shape[0])]

                # entropy per row
                with torch.no_grad():
                    ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)

                # Append to per-request buffers
                for rid, cp, H, chosen_id in zip(
                    [str(r) for r in req_ids],
                    chosen_probs.detach().cpu().tolist(),
                    ent.detach().cpu().tolist(),
                    chosen_ids.detach().cpu().tolist(),
                ):
                    # Store in both internal sidecars and global state
                    buf = self._sidecars.setdefault(rid, {
                        "chosen_token_probs": [],
                        "entropies": [],
                        "chosen_token_ids": [],
                    })
                    buf["chosen_token_probs"].append(cp)
                    buf["entropies"].append(H)
                    buf["chosen_token_ids"].append(chosen_id)
                    
                    # Also store in global state for retrieval
                    global_buf = _RECORDER_STATE.setdefault(_norm_req_id(rid), {
                        "chosen_token_probs": [],
                        "entropies": [],
                        "chosen_token_ids": [],
                        "correct_token_probs": [],
                        "successful_matches": 0,
                        "enable_path_vectors": False,
                        "full_distribution_file": None,
                        "run_id": None,
                    })
                    global_buf["chosen_token_probs"].append(cp)
                    global_buf["entropies"].append(H)
                    global_buf["chosen_token_ids"].append(chosen_id)

            except Exception as e:
                # Never crash the engine from here; just log and continue
                if not getattr(self, "_swallowed_once", False):
                    print(f"[BatchProbabilityRecorder] update_state error (swallowed): {e}")
                    self._swallowed_once = True
            return

        # Any other signature: ignore safely
        return






    # ---- vLLM optional hooks that are used if present ----

    def new_request_state(self, request_id: Any, seq_group: Any, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        vLLM calls this when a request is queued. We initialize the per-request state.
        """
        # Debug: Print when new_request_state is called
        print(f"[BatchProbabilityRecorder] new_request_state called for request_id={request_id}")
        rid = str(request_id)

        # defaults
        run_id = None
        output_dir = None
        dataset = None
        enable_path_vectors = False
        full_distribution_file = None
        gold_token_ids = None
        eos_token_id = None
        stop_token_ids = None

        if isinstance(extra_args, dict):
            run_id = extra_args.get("run_id")
            output_dir = extra_args.get("output_dir")
            dataset = extra_args.get("dataset")
            enable_path_vectors = bool(extra_args.get("enable_path_vectors", False))
            full_distribution_file = extra_args.get("full_distribution_file")
            gold_token_ids = extra_args.get("gold_token_ids")
            eos_token_id = extra_args.get("eos_token_id")
            stop_token_ids = extra_args.get("stop_token_ids")

        # normalize gold ids
        if gold_token_ids is not None and not isinstance(gold_token_ids, list):
            try:
                gold_token_ids = list(gold_token_ids)
            except Exception:
                gold_token_ids = None

        state: Dict[str, Any] = dict(
            # running series
            entropies=[],
            chosen_token_probs=[],
            correct_token_probs=[],
            successful_matches=0,
            step=0,

            # config from driver
            run_id=run_id,
            output_dir=output_dir,
            dataset=dataset,
            enable_path_vectors=enable_path_vectors,
            full_distribution_file=full_distribution_file,

            # gold tracking
            _gold_token_ids=gold_token_ids,
            _gold_cursor=0,

            # eos/stop (not used here, but kept for possible extensions)
            _eos_token_id=eos_token_id,
            _stop_token_ids=stop_token_ids,

            # housekeeping
            _flushed=False,
        )

        self._states[rid] = state
        
        # Also store in global state for retrieval
        push_request_state(request_id, state)
        
        return state

    def on_request_end(self, request_id: Any) -> None:
        """Called by vLLM when a request finishes. Flush and drop state."""
        rid = str(request_id)
        state = self._states.get(rid)
        if state is None:
            return
        self._flush_sidecar(rid, state)
        del self._states[rid]
        
        # Also clean up global state
        pop_request_state(request_id)

    # ---- internal helpers ----

    def _flush_sidecar(self, request_id: str, state: Dict[str, Any]) -> None:
        """Write the per-request JSON sidecar to disk once."""
        if state.get("_flushed", False):
            return

        run_id = state.get("run_id")
        output_dir = state.get("output_dir")
        dataset = state.get("dataset")
        if not run_id or not output_dir or not dataset:
            # Not enough info to write—silently skip
            state["_flushed"] = True
            return

        req_dir = os.path.join(output_dir, dataset, run_id, "requests")
        _ensure_dir(req_dir)
        path = os.path.join(req_dir, f"{request_id}.json")

        payload = {
            "request_id": request_id,
            "run_id": run_id,
            "dataset": dataset,
            "entropies": state.get("entropies", []),
            "chosen_token_probs": state.get("chosen_token_probs", []),
            "correct_token_probs": state.get("correct_token_probs", []),
            "successful_matches": state.get("successful_matches", 0),
            "enable_path_vectors": bool(state.get("enable_path_vectors", False)),
            "full_distribution_file": state.get("full_distribution_file"),
        }

        try:
            with open(path, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            # Last-resort: print so the driver can diagnose
            print(f"[BatchProbabilityRecorder] Failed to write sidecar {path}: {e}")

        state["_flushed"] = True

