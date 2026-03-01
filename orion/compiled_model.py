"""Compilation artifacts: CompiledModel, KeyManifest, EvalKeys.

All three support binary serialization via to_bytes()/from_bytes().
The binary format uses a magic header, JSON metadata, length-prefixed
blobs, and a CRC32 checksum for integrity verification.
"""

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Sequence

from orion.params import CKKSParams, CompilerConfig, CostProfile


# -- Binary format helpers --

_MODEL_MAGIC = b"ORION\x00\x02\x00"
_KEYS_MAGIC = b"ORKEY\x00\x01\x00"


def _pack_container(magic: bytes, metadata: dict, blobs: list[bytes]) -> bytes:
    """Pack metadata + blobs into the standard Orion binary container.

    Format:
        [8 bytes]  MAGIC
        [4 bytes]  METADATA_LENGTH (uint32 LE)
        [N bytes]  METADATA_JSON (utf-8)
        [4 bytes]  BLOB_COUNT (uint32 LE)
        for each blob:
            [8 bytes]  BLOB_LENGTH (uint64 LE)
            [N bytes]  BLOB_DATA
        [4 bytes]  CRC32 of everything above
    """
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    parts = [
        magic,
        struct.pack("<I", len(meta_bytes)),
        meta_bytes,
        struct.pack("<I", len(blobs)),
    ]
    for blob in blobs:
        parts.append(struct.pack("<Q", len(blob)))
        parts.append(blob)
    payload = b"".join(parts)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack("<I", crc)


def _unpack_container(
    data: bytes, expected_magic: bytes
) -> tuple[dict, list[bytes]]:
    """Unpack a binary container, verifying magic and CRC32.

    Returns (metadata_dict, list_of_blobs).
    Raises ValueError on corruption or format mismatch.
    """
    if len(data) < 20:  # 8 magic + 4 meta_len + 4 blob_count + 4 CRC32
        raise ValueError("Data too short to contain a valid container")
    if data[:8] != expected_magic:
        raise ValueError(
            f"Invalid magic: expected {expected_magic!r}, got {data[:8]!r}"
        )

    # Verify CRC32 (last 4 bytes)
    stored_crc = struct.unpack_from("<I", data, len(data) - 4)[0]
    computed_crc = zlib.crc32(data[:-4]) & 0xFFFFFFFF
    if stored_crc != computed_crc:
        raise ValueError(
            f"CRC32 mismatch: stored {stored_crc:#010x}, "
            f"computed {computed_crc:#010x}"
        )

    offset = 8
    (meta_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    meta_json = data[offset : offset + meta_len].decode("utf-8")
    offset += meta_len
    metadata = json.loads(meta_json)

    (blob_count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    blobs = []
    for _ in range(blob_count):
        (blob_len,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        blobs.append(data[offset : offset + blob_len])
        offset += blob_len

    return metadata, blobs


# -- Raw diagonal blob helpers --


def pack_raw_diagonals(diags: dict[int, Sequence[float]], max_slots: int) -> bytes:
    """Pack diagonal vectors into a fixed-stride binary blob.

    Format:
        [4B]                         NUM_DIAGS (uint32 LE)
        [NUM_DIAGS x 4B]             DIAG_INDICES (int32 LE, sorted ascending)
        [NUM_DIAGS x max_slots x 8B] VALUES (float64 LE)

    Each diagonal is zero-padded or truncated to exactly max_slots values.
    """
    indices = sorted(diags.keys())
    num_diags = len(indices)

    parts = [struct.pack("<I", num_diags)]

    for idx in indices:
        parts.append(struct.pack("<i", idx))

    for idx in indices:
        vals = diags[idx]
        padded = list(vals[:max_slots])
        if len(padded) < max_slots:
            padded.extend([0.0] * (max_slots - len(padded)))
        parts.append(struct.pack(f"<{max_slots}d", *padded))

    return b"".join(parts)


def unpack_raw_diagonals(
    data: bytes, max_slots: int
) -> dict[int, list[float]]:
    """Unpack a raw diagonal blob into {diag_index: [float64_values]}.

    Inverse of pack_raw_diagonals().
    """
    offset = 0
    (num_diags,) = struct.unpack_from("<I", data, offset)
    offset += 4

    indices = []
    for _ in range(num_diags):
        (idx,) = struct.unpack_from("<i", data, offset)
        indices.append(idx)
        offset += 4

    result = {}
    for idx in indices:
        vals = list(struct.unpack_from(f"<{max_slots}d", data, offset))
        result[idx] = vals
        offset += max_slots * 8

    return result


def pack_raw_bias(bias: Sequence[float], max_slots: int) -> bytes:
    """Pack a bias vector into a raw float64 blob, zero-padded to max_slots."""
    padded = list(bias[:max_slots])
    if len(padded) < max_slots:
        padded.extend([0.0] * (max_slots - len(padded)))
    return struct.pack(f"<{max_slots}d", *padded)


def unpack_raw_bias(data: bytes, max_slots: int) -> list[float]:
    """Unpack a raw float64 bias blob. Inverse of pack_raw_bias()."""
    return list(struct.unpack(f"<{max_slots}d", data))


# -- Data classes --


@dataclass(frozen=True)
class KeyManifest:
    """Describes which evaluation keys the Client must generate.

    Produced by Compiler.compile(), consumed by Client.generate_keys().
    """

    galois_elements: frozenset[int]
    bootstrap_slots: tuple[int, ...]
    boot_logp: tuple[int, ...] | None
    needs_rlk: bool

    def __post_init__(self):
        if self.bootstrap_slots and self.boot_logp is None:
            raise ValueError(
                "boot_logp must not be None when bootstrap_slots is non-empty"
            )
        # Coerce list inputs
        if isinstance(self.galois_elements, (set, list)):
            object.__setattr__(
                self, "galois_elements", frozenset(self.galois_elements)
            )
        if isinstance(self.bootstrap_slots, list):
            object.__setattr__(
                self, "bootstrap_slots", tuple(self.bootstrap_slots)
            )
        if isinstance(self.boot_logp, list):
            object.__setattr__(self, "boot_logp", tuple(self.boot_logp))

    def to_dict(self) -> dict:
        return {
            "galois_elements": sorted(self.galois_elements),
            "bootstrap_slots": list(self.bootstrap_slots),
            "boot_logp": list(self.boot_logp) if self.boot_logp else None,
            "needs_rlk": self.needs_rlk,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyManifest":
        return cls(
            galois_elements=frozenset(d["galois_elements"]),
            bootstrap_slots=tuple(d["bootstrap_slots"]),
            boot_logp=tuple(d["boot_logp"]) if d.get("boot_logp") else None,
            needs_rlk=d["needs_rlk"],
        )


@dataclass(frozen=True)
class GraphNode:
    """A node in the computation graph.

    Each node represents a single operation in the FHE evaluation pipeline.
    """

    name: str
    op: str
    level: int
    depth: int
    shape: dict | None = None
    config: dict = field(default_factory=dict)
    blob_refs: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "op": self.op,
            "level": self.level,
            "depth": self.depth,
        }
        if self.shape is not None:
            d["shape"] = self.shape
        d["config"] = self.config
        if self.blob_refs is not None:
            d["blob_refs"] = self.blob_refs
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GraphNode":
        return cls(
            name=d["name"],
            op=d["op"],
            level=d["level"],
            depth=d["depth"],
            shape=d.get("shape"),
            config=d.get("config", {}),
            blob_refs=d.get("blob_refs"),
        )


@dataclass(frozen=True)
class GraphEdge:
    """A directed edge in the computation graph."""

    src: str
    dst: str

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst}

    @classmethod
    def from_dict(cls, d: dict) -> "GraphEdge":
        return cls(src=d["src"], dst=d["dst"])


@dataclass(frozen=True)
class Graph:
    """Computation graph: nodes, edges, and input/output identifiers."""

    input: str
    output: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self):
        # Coerce lists to tuples for frozen dataclass
        if isinstance(self.nodes, list):
            object.__setattr__(self, "nodes", tuple(self.nodes))
        if isinstance(self.edges, list):
            object.__setattr__(self, "edges", tuple(self.edges))
        # Validate input/output exist in nodes
        node_names = {n.name for n in self.nodes}
        if self.input not in node_names:
            raise ValueError(
                f"input '{self.input}' not found in graph nodes"
            )
        if self.output not in node_names:
            raise ValueError(
                f"output '{self.output}' not found in graph nodes"
            )
        # Validate all edge endpoints reference existing nodes
        for edge in self.edges:
            if edge.src not in node_names:
                raise ValueError(
                    f"edge src '{edge.src}' not found in graph nodes"
                )
            if edge.dst not in node_names:
                raise ValueError(
                    f"edge dst '{edge.dst}' not found in graph nodes"
                )

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "output": self.output,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        return cls(
            input=d["input"],
            output=d["output"],
            nodes=[GraphNode.from_dict(n) for n in d["nodes"]],
            edges=[GraphEdge.from_dict(e) for e in d["edges"]],
        )


@dataclass
class CompiledModel:
    """Holds all compilation artifacts needed by Client and Evaluator.

    Contains the CKKS params, compiler config, key manifest, computation
    graph (nodes + edges), cost profile, and raw diagonal/bias blobs.
    """

    params: CKKSParams
    config: CompilerConfig
    manifest: KeyManifest
    input_level: int
    cost: CostProfile
    graph: Graph
    blobs: list[bytes]  # binary blobs indexed by node blob_refs

    def to_bytes(self) -> bytes:
        metadata = {
            "version": 2,
            "params": {
                "logn": self.params.logn,
                "logq": list(self.params.logq),
                "logp": list(self.params.logp),
                "logscale": self.params.logscale,
                "h": self.params.h,
                "ring_type": self.params.ring_type,
                "boot_logp": (
                    list(self.params.boot_logp)
                    if self.params.boot_logp
                    else None
                ),
            },
            "config": {
                "margin": self.config.margin,
                "embedding_method": self.config.embedding_method,
                "fuse_modules": self.config.fuse_modules,
            },
            "manifest": self.manifest.to_dict(),
            "input_level": self.input_level,
            "cost": self.cost.to_dict(),
            "graph": self.graph.to_dict(),
            "blob_count": len(self.blobs),
        }
        return _pack_container(_MODEL_MAGIC, metadata, self.blobs)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompiledModel":
        metadata, blobs = _unpack_container(data, _MODEL_MAGIC)

        p = metadata["params"]
        params = CKKSParams(
            logn=p["logn"],
            logq=tuple(p["logq"]),
            logp=tuple(p["logp"]),
            logscale=p["logscale"],
            h=p["h"],
            ring_type=p["ring_type"],
            boot_logp=tuple(p["boot_logp"]) if p.get("boot_logp") else None,
        )

        c = metadata["config"]
        config = CompilerConfig(
            margin=c["margin"],
            embedding_method=c["embedding_method"],
            fuse_modules=c["fuse_modules"],
        )

        manifest = KeyManifest.from_dict(metadata["manifest"])
        cost = CostProfile.from_dict(metadata["cost"])
        graph = Graph.from_dict(metadata["graph"])

        return cls(
            params=params,
            config=config,
            manifest=manifest,
            input_level=metadata["input_level"],
            cost=cost,
            graph=graph,
            blobs=blobs,
        )


@dataclass
class EvalKeys:
    """Serializable container for evaluation keys.

    Holds RLK, per-Galois-element rotation keys, and per-slot-count
    bootstrap keys as raw byte blobs.
    """

    rlk_data: bytes | None = None
    galois_keys: dict[int, bytes] = field(default_factory=dict)
    bootstrap_keys: dict[int, bytes] = field(default_factory=dict)

    @property
    def galois_elements(self) -> set[int]:
        return set(self.galois_keys.keys())

    @property
    def has_rlk(self) -> bool:
        return self.rlk_data is not None

    def to_bytes(self) -> bytes:
        blobs: list[bytes] = []

        rlk_blob_index: int | None = None
        if self.rlk_data is not None:
            rlk_blob_index = len(blobs)
            blobs.append(self.rlk_data)

        galois_index: dict[str, int] = {}
        for gal_el in sorted(self.galois_keys):
            galois_index[str(gal_el)] = len(blobs)
            blobs.append(self.galois_keys[gal_el])

        bootstrap_index: dict[str, int] = {}
        for slots in sorted(self.bootstrap_keys):
            bootstrap_index[str(slots)] = len(blobs)
            blobs.append(self.bootstrap_keys[slots])

        metadata = {
            "version": 1,
            "rlk_blob_index": rlk_blob_index,
            "galois_keys": galois_index,
            "bootstrap_keys": bootstrap_index,
            "blob_count": len(blobs),
        }
        return _pack_container(_KEYS_MAGIC, metadata, blobs)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvalKeys":
        metadata, blobs = _unpack_container(data, _KEYS_MAGIC)

        rlk_data = None
        if metadata["rlk_blob_index"] is not None:
            rlk_data = blobs[metadata["rlk_blob_index"]]

        galois_keys = {}
        for gal_el_str, idx in metadata["galois_keys"].items():
            galois_keys[int(gal_el_str)] = blobs[idx]

        bootstrap_keys = {}
        for slots_str, idx in metadata["bootstrap_keys"].items():
            bootstrap_keys[int(slots_str)] = blobs[idx]

        return cls(
            rlk_data=rlk_data,
            galois_keys=galois_keys,
            bootstrap_keys=bootstrap_keys,
        )
