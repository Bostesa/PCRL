"""WIDE matched-control deployment on original global source-bank codes.

No labels, person identifiers or training data are runtime inputs. Internal
KEEP and named-code aliases stay internal; all laws emit original global IDs.
Fitting and resource reservations belong to the separately queued worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path

import numpy as np


def _weights(value, width):
    array = np.asarray(value, np.float64)
    if (array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all()
            or np.any(array < 0) or not np.allclose(array.sum(1), 1., atol=1e-12, rtol=0)):
        raise ValueError("WIDE marginal weights must be normalized finite tables")
    array = array.copy(); array.setflags(write=False)
    return array


@dataclass
class ControlWidePolicy:
    space: object
    kind: str
    context_map: object
    first_weights: np.ndarray
    second_weights: object = None
    second_keys: object = None
    first_recipient: int = 1
    metadata: object = None

    def __post_init__(self):
        if self.kind not in ("LOCAL", "PRODUCT", "DETERMINISTIC", "SEQUENTIAL"):
            raise ValueError("unregistered WIDE control kind")
        self._classes = (self.space.probability1.argmax(1), self.space.probability2.argmax(1))
        self._axes = tuple({int(k): np.r_[-1, np.flatnonzero(predicted == k)].astype(np.int64)
                            for k in np.unique(predicted)} for predicted in self._classes)
        self._widths = tuple(max(map(len, axes.values())) for axes in self._axes)
        if self.kind == "LOCAL":
            # Context objects are deliberately neither inspected nor applied.
            self.first_weights = _weights(self.first_weights, self._widths[0])
            self.second_weights = _weights(self.second_weights, self._widths[1])
            if (len(self.first_weights) != len(self._classes[0])
                    or len(self.second_weights) != len(self._classes[1])):
                raise ValueError("WIDE local tables must cover each complete own-code bank")
            for predicted, axes, table in zip(self._classes, self._axes, (self.first_weights, self.second_weights)):
                for k, axis in axes.items():
                    if np.any(table[predicted == k, len(axis):] != 0):
                        raise ValueError("WIDE local padding cannot carry mass")
            self.metadata = None
            return
        if self.context_map is None or not isinstance(self.metadata, dict):
            raise ValueError("central WIDE control needs frozen context and template descriptors")
        self.metadata = copy.deepcopy(self.metadata)
        if self.metadata.get("schema") != "CBR1-context-template-v1" or self.metadata.get("support") != "WIDE":
            raise ValueError("WIDE control descriptor schema differs")
        descriptions = self.metadata["contexts"]
        groups = len(self.context_map.labels)
        if len(descriptions) != groups or self.metadata["supported_contexts"] != groups:
            raise ValueError("WIDE descriptors do not cover frozen contexts")
        offset = 0; maximum = 1
        for g, desc in enumerate(descriptions):
            if desc["context"] != g or len(desc["class_pair"]) != 2:
                raise ValueError("WIDE context descriptor order differs")
            for r, key in enumerate(("recipient1_templates", "recipient2_templates")):
                k = desc["class_pair"][r]
                if k not in self._axes[r] or desc[key] != self._axes[r][k].tolist():
                    raise ValueError("WIDE axis must be KEEP then every same-class original ID in order")
            count = len(desc["recipient1_templates"])*len(desc["recipient2_templates"])
            if desc["action_count"] != count or desc["parameter_offset"] != offset:
                raise ValueError("WIDE Cartesian parameter offsets differ")
            offset += count; maximum = max(maximum, count)
        if self.metadata["parameters"] != max(1, offset) or self.metadata["maximum_actions_per_row"] != maximum:
            raise ValueError("WIDE descriptor dimensions differ")
        if self.kind == "DETERMINISTIC":
            weights = np.asarray(self.first_weights, np.float64)
            if weights.shape != (self.metadata["parameters"],) or not np.isin(weights, [0., 1.]).all():
                raise ValueError("WIDE deterministic deployment requires exact binary flattened weights")
            for desc in descriptions:
                start, count = desc["parameter_offset"], desc["action_count"]
                if weights[start:start+count].sum() != 1.:
                    raise ValueError("each WIDE deterministic context needs exactly one action")
            if not descriptions and np.any(weights):
                raise ValueError("identity-only descriptor has no free actions")
            self.first_weights = weights.copy(); self.first_weights.setflags(write=False)
            return
        first_r = self.first_recipient-1 if self.kind == "SEQUENTIAL" else 0
        if self.first_recipient not in (1, 2):
            raise ValueError("WIDE sequential first recipient differs")
        second_r = 1-first_r
        self.first_weights = _weights(self.first_weights, self._widths[first_r])
        self.second_weights = _weights(self.second_weights, self._widths[second_r])
        if len(self.first_weights) != groups:
            raise ValueError("WIDE first marginal context table differs")
        for g, desc in enumerate(descriptions):
            width = len(self._axes[first_r][desc["class_pair"][first_r]])
            if np.any(self.first_weights[g, width:] != 0):
                raise ValueError("WIDE first marginal padding cannot carry mass")
        if self.kind == "PRODUCT":
            if len(self.second_weights) != groups:
                raise ValueError("WIDE product second marginal context table differs")
            self._second_lookup = None
            second_contexts = np.arange(groups)
        else:
            keys = np.asarray(self.second_keys)
            if (keys.dtype != np.int64 or keys.shape != (len(self.second_weights), 2)
                    or len(keys) > 512 or np.any(keys[:, 0] < 0) or np.any(keys[:, 0] >= groups)
                    or np.any(keys[:, 1] < -1) or len(set(map(tuple, keys))) != len(keys)):
                raise ValueError("WIDE sequential key table differs")
            self._second_lookup = {tuple(map(int, key)): i for i, key in enumerate(keys)}
            for g in range(groups):
                if (g, -1) not in self._second_lookup:
                    raise ValueError("each WIDE sequential context needs a fallback parent")
            for g, code in keys:
                if code >= 0 and code not in self._axes[first_r][descriptions[g]["class_pair"][first_r]][1:]:
                    raise ValueError("sequential child must name an actual same-class emitted-first global ID")
            self.second_keys = keys.copy(); self.second_keys.setflags(write=False)
            second_contexts = keys[:, 0]
        for i, g in enumerate(second_contexts):
            width = len(self._axes[second_r][descriptions[g]["class_pair"][second_r]])
            if np.any(self.second_weights[i, width:] != 0):
                raise ValueError("WIDE second marginal padding cannot carry mass")

    def _inputs(self, c1, c2):
        c1, c2 = np.asarray(c1), np.asarray(c2)
        if (c1.dtype != np.int64 or c2.dtype != np.int64 or c1.ndim != 1 or c2.shape != c1.shape
                or np.any(c1 < 0) or np.any(c1 >= len(self._classes[0]))
                or np.any(c2 < 0) or np.any(c2 >= len(self._classes[1]))):
            raise ValueError("WIDE runtime needs aligned valid int64 original source codes")
        return c1, c2

    def law_codes(self, c1, c2):
        c1, c2 = self._inputs(c1, c2); n = len(c1)
        width = self._widths[0]*self._widths[1] if self.kind == "LOCAL" else self.metadata["maximum_actions_per_row"]
        emitted1 = np.repeat(c1[:, None], width, axis=1)
        emitted2 = np.repeat(c2[:, None], width, axis=1)
        mass = np.zeros((n, width), np.float64); mass[:, 0] = 1.
        if not n: return emitted1, emitted2, mass
        if self.kind == "LOCAL":
            for k1, axis1 in self._axes[0].items():
                for k2, axis2 in self._axes[1].items():
                    selected = np.flatnonzero((self._classes[0][c1] == k1) & (self._classes[1][c2] == k2))
                    self._populate(selected, c1, c2, axis1, axis2, emitted1, emitted2, mass,
                                   (self.first_weights[c1[selected], :len(axis1), None]
                                    *self.second_weights[c2[selected], None, :len(axis2)]).reshape(len(selected), len(axis1)*len(axis2)))
        else:
            contexts = self.context_map.apply(self.space.probability1[c1], self.space.probability2[c2])
            if np.asarray(contexts).dtype != np.int64 or np.asarray(contexts).shape != c1.shape or np.any(contexts < -1) or np.any(contexts >= len(self.metadata["contexts"])):
                raise ValueError("WIDE frozen context routing changed")
            for desc in self.metadata["contexts"]:
                g = desc["context"]; selected = np.flatnonzero(contexts == g)
                if not len(selected): continue
                if any(np.any(self._classes[r][(c1, c2)[r][selected]] != desc["class_pair"][r]) for r in (0, 1)):
                    raise ValueError("WIDE frozen context changed predicted class pair")
                axis1, axis2 = (np.asarray(desc[key], np.int64) for key in ("recipient1_templates", "recipient2_templates"))
                if self.kind == "DETERMINISTIC":
                    offset, count = desc["parameter_offset"], desc["action_count"]
                    row_mass = np.broadcast_to(self.first_weights[offset:offset+count], (len(selected), count))
                elif self.kind == "PRODUCT":
                    row_mass = np.broadcast_to((self.first_weights[g, :len(axis1), None]
                                               *self.second_weights[g, None, :len(axis2)]).ravel(),
                                              (len(selected), len(axis1)*len(axis2)))
                else:
                    first_r = self.first_recipient-1
                    first_axis, second_axis = (axis1, axis2) if first_r == 0 else (axis2, axis1)
                    first_codes = np.broadcast_to(first_axis, (len(selected), len(first_axis))).copy()
                    first_codes[:, 0] = (c1, c2)[first_r][selected]
                    parent = self._second_lookup[(g, -1)]
                    groups = np.array([[self._second_lookup.get((g, int(code)), parent) for code in row]
                                       for row in first_codes], np.int64)
                    q = self.first_weights[g, :len(first_axis)][None, :, None]*self.second_weights[groups, :len(second_axis)]
                    row_mass = (q if first_r == 0 else q.transpose(0, 2, 1)).reshape(len(selected), len(axis1)*len(axis2))
                self._populate(selected, c1, c2, axis1, axis2, emitted1, emitted2, mass, row_mass)
        if not np.allclose(mass.sum(1), 1., atol=1e-12, rtol=0):
            raise ValueError("WIDE deployed mass is not normalized")
        return emitted1, emitted2, mass

    @staticmethod
    def _populate(selected, c1, c2, axis1, axis2, emitted1, emitted2, mass, row_mass):
        if not len(selected): return
        a, b = np.repeat(axis1, len(axis2)), np.tile(axis2, len(axis1))
        aa = np.broadcast_to(a, (len(selected), len(a))).copy()
        bb = np.broadcast_to(b, (len(selected), len(b))).copy()
        aa[:, a == -1] = c1[selected, None]; bb[:, b == -1] = c2[selected, None]
        emitted1[selected, :len(a)] = aa; emitted2[selected, :len(b)] = bb
        mass[selected] = 0.; mass[selected, :len(a)] = row_mass

    def law_prepared(self, p):
        """WIDE mass, aligned to law_codes, never the inherited16-action law."""
        return self.law_codes(p["b0_code_r1"], p["b0_code_r2"])[2]


def save_wide_control_policy(directory, policy, *, seed, arm_id, budget, source_manifest):
    from .fit import write_npz
    from .source import write_once
    directory = Path(directory)
    if directory.exists() or ".private" not in directory.parts:
        raise PermissionError("new private WIDE control package required")
    directory.mkdir(parents=True, mode=0o700)
    arrays = {"first_weights": policy.first_weights}
    if policy.kind != "DETERMINISTIC": arrays["second_weights"] = policy.second_weights
    if policy.kind == "SEQUENTIAL": arrays["second_keys"] = policy.second_keys
    context = None if policy.kind == "LOCAL" else policy.context_map
    if context is not None:
        arrays.update(context_mapping=context.mapping, raw_fit_counts=context.raw_fit_counts,
                      merged_fit_counts=context.merged_fit_counts)
    package_sha = write_npz(directory/"control.npz", arrays)
    record = {"schema": "CBR1-wide-control-policy-v1", "support": "WIDE", "seed": int(seed),
              "arm_id": arm_id, "budget": budget, "kind": policy.kind,
              "first_recipient": policy.first_recipient, "package_sha256": package_sha,
              "context_labels": None if context is None else list(context.labels), "metadata": policy.metadata,
              "public_bank_sha256": source_manifest["seeds"][str(seed)]["public_bank"]["sha256"],
              "parameter_count": int(policy.first_weights.size + (0 if policy.kind == "DETERMINISTIC" else policy.second_weights.size)),
              "runtime_sensitive_or_task_labels": False, "public_wire": "original code tuples and frozen source metadata",
              "locality": "own P0 code only, independent coins" if policy.kind == "LOCAL" else "central P0 inputs"}
    return record, write_once(directory/"POLICY_RECEIPT.json", record)


def load_wide_control_policy(directory, manifest, *, expected_receipt_sha256=None):
    from .source import sha_file, read_exact, load_historical_policy
    from .contexts import ContextMap
    directory = Path(directory); receipt = directory/"POLICY_RECEIPT.json"
    if expected_receipt_sha256 and sha_file(receipt) != expected_receipt_sha256:
        raise PermissionError("WIDE control receipt differs")
    record = json.loads(receipt.read_text())
    if record["schema"] != "CBR1-wide-control-policy-v1" or record["support"] != "WIDE":
        raise ValueError("wrong WIDE control package schema")
    if record["public_bank_sha256"] != manifest["seeds"][str(record["seed"])]["public_bank"]["sha256"]:
        raise PermissionError("WIDE control bank source differs")
    keys = {"first_weights"}
    if record["kind"] != "DETERMINISTIC": keys.add("second_weights")
    if record["kind"] == "SEQUENTIAL": keys.add("second_keys")
    if record["context_labels"] is not None: keys.update({"context_mapping", "raw_fit_counts", "merged_fit_counts"})
    arrays = read_exact({"path": str(directory/"control.npz"), "sha256": record["package_sha256"]}, keys)
    context = None if record["context_labels"] is None else ContextMap(
        arrays["context_mapping"], arrays["raw_fit_counts"], arrays["merged_fit_counts"], tuple(record["context_labels"]))
    space = load_historical_policy(manifest, record["seed"], "V_J").space
    policy = ControlWidePolicy(space, record["kind"], context, arrays["first_weights"], arrays.get("second_weights"),
                               arrays.get("second_keys"), record["first_recipient"], record["metadata"])
    count = policy.first_weights.size + (0 if policy.kind == "DETERMINISTIC" else policy.second_weights.size)
    if count != record["parameter_count"]: raise PermissionError("WIDE control parameter count differs")
    return policy
