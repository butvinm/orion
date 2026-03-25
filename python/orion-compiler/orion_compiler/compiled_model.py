"""Compilation artifacts: CompiledModel, KeyManifest.

Both support binary serialization via to_bytes()/from_bytes().
The binary format uses a magic header, JSON metadata, and length-prefixed blobs.
"""

import json
import os
import struct
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from orion_compiler.params import CKKSParams, CompilerConfig, CostProfile


class BlobStore:
    """Write-once blob storage backed by a temporary file.

    Blobs are written to disk as they're appended, keeping memory usage
    proportional to one blob at a time. Implements __len__ and __getitem__
    so it can be used anywhere list[bytes] is expected for reading.
    """

    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile()  # noqa: SIM115
        self._entries: list[tuple[int, int]] = []  # (offset, length)

    def append(self, data: bytes) -> int:
        """Append a blob, return its index."""
        offset = self._file.tell()
        self._file.write(data)
        idx = len(self._entries)
        self._entries.append((offset, len(data)))
        return idx

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> bytes:
        offset, length = self._entries[idx]
        self._file.seek(offset)
        return self._file.read(length)

    def __iter__(self) -> Iterator[bytes]:
        for i in range(len(self._entries)):
            yield self[i]

    def close(self) -> None:
        self._file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# -- Binary format helpers --

_MODEL_MAGIC = b"ORION\x00\x02\x00"


def _pack_container(magic: bytes, metadata: dict[str, Any], blobs: list[bytes]) -> bytes:
    """Pack metadata + blobs into the standard Orion binary container.

    Format:
        [8 bytes]  MAGIC
        [4 bytes]  METADATA_LENGTH (uint32 LE)
        [N bytes]  METADATA_JSON (utf-8)
        [4 bytes]  BLOB_COUNT (uint32 LE)
        for each blob:
            [8 bytes]  BLOB_LENGTH (uint64 LE)
            [N bytes]  BLOB_DATA
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
    return b"".join(parts)


def _unpack_container(data: bytes, expected_magic: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """Unpack a binary container, verifying magic.

    Returns (metadata_dict, list_of_blobs).
    Raises ValueError on corruption or format mismatch.
    """
    if len(data) < 16:  # 8 magic + 4 meta_len + 4 blob_count
        raise ValueError("Data too short to contain a valid container")
    if data[:8] != expected_magic:
        raise ValueError(f"Invalid magic: expected {expected_magic!r}, got {data[:8]!r}")

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


def unpack_raw_diagonals(data: bytes, max_slots: int) -> dict[int, list[float]]:
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
    btp_logn: int | None
    needs_rlk: bool

    def __post_init__(self) -> None:
        if self.bootstrap_slots and self.boot_logp is None:
            raise ValueError("boot_logp must not be None when bootstrap_slots is non-empty")
        # Coerce list inputs
        if isinstance(self.galois_elements, (set, list)):
            object.__setattr__(self, "galois_elements", frozenset(self.galois_elements))
        if isinstance(self.bootstrap_slots, list):
            object.__setattr__(self, "bootstrap_slots", tuple(self.bootstrap_slots))
        if isinstance(self.boot_logp, list):
            object.__setattr__(self, "boot_logp", tuple(self.boot_logp))

    def to_dict(self) -> dict[str, Any]:
        return {
            "galois_elements": sorted(self.galois_elements),
            "bootstrap_slots": list(self.bootstrap_slots),
            "boot_logp": list(self.boot_logp) if self.boot_logp else None,
            "btp_logn": self.btp_logn,
            "needs_rlk": self.needs_rlk,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KeyManifest":
        return cls(
            galois_elements=frozenset(d["galois_elements"]),
            bootstrap_slots=tuple(d["bootstrap_slots"]),
            boot_logp=tuple(d["boot_logp"]) if d.get("boot_logp") else None,
            btp_logn=d.get("btp_logn"),
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
    shape: dict[str, list[int]] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    blob_refs: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
    def from_dict(cls, d: dict[str, Any]) -> "GraphNode":
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

    def to_dict(self) -> dict[str, str]:
        return {"src": self.src, "dst": self.dst}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "GraphEdge":
        return cls(src=d["src"], dst=d["dst"])


@dataclass(frozen=True)
class Graph:
    """Computation graph: nodes, edges, and input/output identifiers."""

    input: str
    output: str
    nodes: tuple[GraphNode, ...] | list[GraphNode]
    edges: tuple[GraphEdge, ...] | list[GraphEdge]

    def __post_init__(self) -> None:
        # Coerce lists to tuples for frozen dataclass
        if isinstance(self.nodes, list):
            object.__setattr__(self, "nodes", tuple(self.nodes))
        if isinstance(self.edges, list):
            object.__setattr__(self, "edges", tuple(self.edges))
        # Validate input/output exist in nodes
        node_names = {n.name for n in self.nodes}
        if self.input not in node_names:
            raise ValueError(f"input '{self.input}' not found in graph nodes")
        if self.output not in node_names:
            raise ValueError(f"output '{self.output}' not found in graph nodes")
        # Validate all edge endpoints reference existing nodes
        for edge in self.edges:
            if edge.src not in node_names:
                raise ValueError(f"edge src '{edge.src}' not found in graph nodes")
            if edge.dst not in node_names:
                raise ValueError(f"edge dst '{edge.dst}' not found in graph nodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Graph":
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
    blobs: list[bytes] | BlobStore  # binary blobs indexed by node blob_refs

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "version": 2,
            "params": {
                "logn": self.params.logn,
                "logq": list(self.params.logq),
                "logp": list(self.params.logp),
                "log_default_scale": self.params.log_default_scale,
                "h": self.params.h,
                "ring_type": self.params.ring_type,
                "boot_logp": (list(self.params.boot_logp) if self.params.boot_logp else None),
                "btp_logn": self.params.btp_logn,
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

    def to_bytes(self) -> bytes:
        return _pack_container(_MODEL_MAGIC, self._build_metadata(), list(self.blobs))

    def to_file(self, path: str | os.PathLike[str]) -> None:
        """Write the compiled model to a file, streaming blobs one at a time.

        Unlike to_bytes(), this avoids creating a single large bytes object,
        reducing peak memory by the size of the .orion file.
        """
        metadata = self._build_metadata()
        meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(_MODEL_MAGIC)
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack("<I", len(self.blobs)))
            for blob in self.blobs:
                f.write(struct.pack("<Q", len(blob)))
                f.write(blob)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompiledModel":
        metadata, blobs = _unpack_container(data, _MODEL_MAGIC)

        version = metadata.get("version")
        if version != 2:
            raise ValueError(f"Unsupported format version {version} (expected 2)")

        p = metadata["params"]
        params = CKKSParams(
            logn=p["logn"],
            logq=tuple(p["logq"]),
            logp=tuple(p["logp"]),
            log_default_scale=p["log_default_scale"],
            h=p["h"],
            ring_type=p["ring_type"],
            boot_logp=tuple(p["boot_logp"]) if p.get("boot_logp") else None,
            btp_logn=p.get("btp_logn"),
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
