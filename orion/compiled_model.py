"""Compilation artifacts: CompiledModel, KeyManifest, EvalKeys.

All three support binary serialization via to_bytes()/from_bytes().
The binary format uses a magic header, JSON metadata, length-prefixed
blobs, and a CRC32 checksum for integrity verification.
"""

import json
import struct
import zlib
from dataclasses import dataclass, field

from orion.params import CKKSParams, CompilerConfig


# -- Binary format helpers --

_MODEL_MAGIC = b"ORMDL\x00\x01\x00"
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
    if len(data) < 8:
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


@dataclass
class CompiledModel:
    """Holds all compilation artifacts needed by Client and Evaluator.

    Contains the CKKS params, compiler config, key manifest, per-module
    metadata, execution topology, and serialized LinearTransform blobs.
    """

    params: CKKSParams
    config: CompilerConfig
    manifest: KeyManifest
    input_level: int
    module_metadata: dict  # module_name -> metadata dict
    topology: list[str]  # execution order
    blobs: list[bytes]  # binary blobs indexed by module metadata

    def to_bytes(self) -> bytes:
        metadata = {
            "version": 1,
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
            "modules": self.module_metadata,
            "topology": self.topology,
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

        return cls(
            params=params,
            config=config,
            manifest=manifest,
            input_level=metadata["input_level"],
            module_metadata=metadata["modules"],
            topology=metadata["topology"],
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
        blob_idx = 0

        rlk_blob_index: int | None = None
        if self.rlk_data is not None:
            rlk_blob_index = blob_idx
            blobs.append(self.rlk_data)
            blob_idx += 1

        galois_index: dict[str, int] = {}
        for gal_el in sorted(self.galois_keys):
            galois_index[str(gal_el)] = blob_idx
            blobs.append(self.galois_keys[gal_el])
            blob_idx += 1

        bootstrap_index: dict[str, int] = {}
        for slots in sorted(self.bootstrap_keys):
            bootstrap_index[str(slots)] = blob_idx
            blobs.append(self.bootstrap_keys[slots])
            blob_idx += 1

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
