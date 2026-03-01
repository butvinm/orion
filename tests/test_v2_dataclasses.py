"""Unit tests for v2 data structures: CKKSParams, CompilerConfig,
KeyManifest, CompiledModel, EvalKeys, and the NewParameters adapter."""

import pytest

from orion.params import CKKSParams, CostProfile, CompilerConfig
from orion.compiled_model import (
    CompiledModel,
    KeyManifest,
    EvalKeys,
    Graph,
    GraphNode,
    GraphEdge,
)
from orion.core.compiler_backend import NewParameters


# ---------------------------------------------------------------------------
# CKKSParams
# ---------------------------------------------------------------------------


class TestCKKSParams:
    def _default_params(self, **overrides):
        defaults = dict(
            logn=14,
            logq=(55, 40, 40, 40),
            logp=(61, 61),
            logscale=40,
        )
        defaults.update(overrides)
        return CKKSParams(**defaults)

    def test_basic_construction(self):
        p = self._default_params()
        assert p.logn == 14
        assert p.logq == (55, 40, 40, 40)
        assert p.logp == (61, 61)
        assert p.logscale == 40
        assert p.h == 192
        assert p.ring_type == "conjugate_invariant"
        assert p.boot_logp is None

    def test_list_inputs_coerced_to_tuples(self):
        p = CKKSParams(logn=14, logq=[55, 40], logp=[61], logscale=40)
        assert isinstance(p.logq, tuple)
        assert isinstance(p.logp, tuple)

    def test_boot_logp_list_coerced(self):
        p = self._default_params(boot_logp=[61, 61])
        assert isinstance(p.boot_logp, tuple)

    def test_max_level(self):
        p = self._default_params()
        assert p.max_level == 3  # len((55, 40, 40, 40)) - 1

    def test_max_slots_conjugate_invariant(self):
        p = self._default_params(ring_type="conjugate_invariant")
        assert p.max_slots == 2**14

    def test_max_slots_standard(self):
        p = self._default_params(ring_type="standard")
        assert p.max_slots == 2**13

    def test_ring_degree(self):
        p = self._default_params()
        assert p.ring_degree == 2**14

    def test_invalid_logn(self):
        with pytest.raises(ValueError, match="logn must be positive"):
            self._default_params(logn=0)

    def test_empty_logq(self):
        with pytest.raises(ValueError, match="logq must be non-empty"):
            self._default_params(logq=())

    def test_empty_logp(self):
        with pytest.raises(ValueError, match="logp must be non-empty"):
            self._default_params(logp=())

    def test_logp_longer_than_logq(self):
        with pytest.raises(ValueError, match="logp length"):
            self._default_params(logq=(55,), logp=(61, 61))

    def test_invalid_ring_type(self):
        with pytest.raises(ValueError, match="ring_type must be one of"):
            self._default_params(ring_type="invalid")


# ---------------------------------------------------------------------------
# CompilerConfig
# ---------------------------------------------------------------------------


class TestCompilerConfig:
    def test_defaults(self):
        c = CompilerConfig()
        assert c.margin == 2
        assert c.embedding_method == "hybrid"
        assert c.fuse_modules is True

    def test_custom(self):
        c = CompilerConfig(margin=3, embedding_method="square", fuse_modules=False)
        assert c.margin == 3
        assert c.embedding_method == "square"
        assert c.fuse_modules is False

    def test_invalid_embedding_method(self):
        with pytest.raises(ValueError, match="embedding_method must be one of"):
            CompilerConfig(embedding_method="invalid")


# ---------------------------------------------------------------------------
# KeyManifest
# ---------------------------------------------------------------------------


class TestKeyManifest:
    def test_basic_construction(self):
        m = KeyManifest(
            galois_elements=frozenset({5, 25, 125}),
            bootstrap_slots=(4096,),
            boot_logp=(61, 61),
            needs_rlk=True,
        )
        assert m.galois_elements == frozenset({5, 25, 125})
        assert m.bootstrap_slots == (4096,)
        assert m.needs_rlk is True

    def test_no_bootstrap(self):
        m = KeyManifest(
            galois_elements=frozenset({5}),
            bootstrap_slots=(),
            boot_logp=None,
            needs_rlk=False,
        )
        assert m.bootstrap_slots == ()
        assert m.boot_logp is None

    def test_bootstrap_without_boot_logp_raises(self):
        with pytest.raises(ValueError, match="boot_logp must not be None"):
            KeyManifest(
                galois_elements=frozenset(),
                bootstrap_slots=(4096,),
                boot_logp=None,
                needs_rlk=True,
            )

    def test_list_inputs_coerced(self):
        m = KeyManifest(
            galois_elements={5, 25},
            bootstrap_slots=[4096],
            boot_logp=[61],
            needs_rlk=True,
        )
        assert isinstance(m.galois_elements, frozenset)
        assert isinstance(m.bootstrap_slots, tuple)
        assert isinstance(m.boot_logp, tuple)

    def test_roundtrip_dict(self):
        m = KeyManifest(
            galois_elements=frozenset({5, 25, 125}),
            bootstrap_slots=(4096,),
            boot_logp=(61, 61),
            needs_rlk=True,
        )
        d = m.to_dict()
        m2 = KeyManifest.from_dict(d)
        assert m2.galois_elements == m.galois_elements
        assert m2.bootstrap_slots == m.bootstrap_slots
        assert m2.boot_logp == m.boot_logp
        assert m2.needs_rlk == m.needs_rlk


# ---------------------------------------------------------------------------
# CompiledModel
# ---------------------------------------------------------------------------


def _sample_graph():
    nodes = [
        GraphNode(name="fc1", op="linear_transform", level=2, depth=1,
                  config={"bsgs_ratio": 2.0, "output_rotations": 0},
                  blob_refs={"diag_0_0": 0, "bias": 1}),
        GraphNode(name="act1", op="polynomial", level=1, depth=2,
                  config={"coeffs": [0.5, 0.1], "basis": "chebyshev",
                          "prescale": 1, "postscale": 1, "constant": 0}),
    ]
    edges = [GraphEdge(src="fc1", dst="act1")]
    return Graph(input="fc1", output="act1", nodes=nodes, edges=edges)


def _sample_compiled_model():
    params = CKKSParams(logn=14, logq=(55, 40, 40), logp=(61,), logscale=40)
    config = CompilerConfig()
    manifest = KeyManifest(
        galois_elements=frozenset({5, 25}),
        bootstrap_slots=(),
        boot_logp=None,
        needs_rlk=True,
    )
    cost = CostProfile(bootstrap_count=0, galois_key_count=2, bootstrap_key_count=0)
    graph = _sample_graph()
    blobs = [b"blob_for_fc1_transform", b"blob_for_fc1_bias", b""]
    return CompiledModel(
        params=params,
        config=config,
        manifest=manifest,
        input_level=2,
        cost=cost,
        graph=graph,
        blobs=blobs,
    )


class TestCompiledModel:
    def test_construction(self):
        cm = _sample_compiled_model()
        assert cm.input_level == 2
        assert len(cm.blobs) == 3
        assert len(cm.graph.nodes) == 2
        assert len(cm.graph.edges) == 1

    def test_serialization_roundtrip(self):
        cm = _sample_compiled_model()
        data = cm.to_bytes()
        cm2 = CompiledModel.from_bytes(data)

        assert cm2.params.logn == cm.params.logn
        assert cm2.params.logq == cm.params.logq
        assert cm2.params.logp == cm.params.logp
        assert cm2.params.logscale == cm.params.logscale
        assert cm2.params.h == cm.params.h
        assert cm2.params.ring_type == cm.params.ring_type
        assert cm2.config.margin == cm.config.margin
        assert cm2.config.embedding_method == cm.config.embedding_method
        assert cm2.manifest.galois_elements == cm.manifest.galois_elements
        assert cm2.manifest.needs_rlk == cm.manifest.needs_rlk
        assert cm2.input_level == cm.input_level
        assert cm2.cost.bootstrap_count == cm.cost.bootstrap_count
        assert cm2.cost.galois_key_count == cm.cost.galois_key_count
        assert cm2.graph.input == cm.graph.input
        assert cm2.graph.output == cm.graph.output
        assert len(cm2.graph.nodes) == len(cm.graph.nodes)
        assert len(cm2.graph.edges) == len(cm.graph.edges)
        assert cm2.blobs == cm.blobs

    def test_wrong_magic(self):
        cm = _sample_compiled_model()
        data = bytearray(cm.to_bytes())
        data[:8] = b"BADMAGIC"
        with pytest.raises(ValueError, match="Invalid magic"):
            CompiledModel.from_bytes(bytes(data))

    def test_empty_blobs(self):
        params = CKKSParams(logn=14, logq=(55, 40), logp=(61,), logscale=40)
        graph = Graph(
            input="x", output="x",
            nodes=[GraphNode(name="x", op="flatten", level=0, depth=0)],
            edges=[],
        )
        cm = CompiledModel(
            params=params,
            config=CompilerConfig(),
            manifest=KeyManifest(
                galois_elements=frozenset(),
                bootstrap_slots=(),
                boot_logp=None,
                needs_rlk=False,
            ),
            input_level=1,
            cost=CostProfile(bootstrap_count=0, galois_key_count=0,
                             bootstrap_key_count=0),
            graph=graph,
            blobs=[],
        )
        data = cm.to_bytes()
        cm2 = CompiledModel.from_bytes(data)
        assert cm2.blobs == []
        assert len(cm2.graph.nodes) == 1

    def test_large_blob(self):
        """Verify blobs up to ~1MB roundtrip correctly."""
        params = CKKSParams(logn=14, logq=(55, 40), logp=(61,), logscale=40)
        big_blob = bytes(range(256)) * 4096  # ~1MB
        graph = Graph(
            input="x", output="x",
            nodes=[GraphNode(name="x", op="flatten", level=0, depth=0)],
            edges=[],
        )
        cm = CompiledModel(
            params=params,
            config=CompilerConfig(),
            manifest=KeyManifest(
                galois_elements=frozenset(),
                bootstrap_slots=(),
                boot_logp=None,
                needs_rlk=False,
            ),
            input_level=1,
            cost=CostProfile(bootstrap_count=0, galois_key_count=0,
                             bootstrap_key_count=0),
            graph=graph,
            blobs=[big_blob],
        )
        data = cm.to_bytes()
        cm2 = CompiledModel.from_bytes(data)
        assert cm2.blobs[0] == big_blob


# ---------------------------------------------------------------------------
# EvalKeys
# ---------------------------------------------------------------------------


class TestEvalKeys:
    def test_empty_keys(self):
        ek = EvalKeys()
        assert not ek.has_rlk
        assert ek.galois_elements == set()

    def test_with_rlk(self):
        ek = EvalKeys(rlk_data=b"rlk_bytes")
        assert ek.has_rlk

    def test_properties(self):
        ek = EvalKeys(
            rlk_data=b"rlk",
            galois_keys={5: b"gk5", 25: b"gk25"},
            bootstrap_keys={4096: b"bk4096"},
        )
        assert ek.galois_elements == {5, 25}
        assert ek.has_rlk is True

    def test_serialization_roundtrip(self):
        ek = EvalKeys(
            rlk_data=b"relinearization_key_data",
            galois_keys={5: b"galois_5", 25: b"galois_25", 125: b"galois_125"},
            bootstrap_keys={4096: b"bootstrap_4096"},
        )
        data = ek.to_bytes()
        ek2 = EvalKeys.from_bytes(data)

        assert ek2.rlk_data == ek.rlk_data
        assert ek2.galois_keys == ek.galois_keys
        assert ek2.bootstrap_keys == ek.bootstrap_keys

    def test_serialization_no_rlk(self):
        ek = EvalKeys(
            galois_keys={5: b"gk5"},
        )
        data = ek.to_bytes()
        ek2 = EvalKeys.from_bytes(data)
        assert ek2.rlk_data is None
        assert ek2.galois_keys == {5: b"gk5"}

    def test_wrong_magic(self):
        ek = EvalKeys(rlk_data=b"rlk")
        data = bytearray(ek.to_bytes())
        data[:8] = b"ORMDL\x00\x01\x00"  # model magic instead of keys magic
        with pytest.raises(ValueError, match="Invalid magic"):
            EvalKeys.from_bytes(bytes(data))


# ---------------------------------------------------------------------------
# NewParameters.from_ckks_params adapter
# ---------------------------------------------------------------------------


class TestNewParametersAdapter:
    def test_from_ckks_params_basic(self):
        ckks = CKKSParams(
            logn=14,
            logq=(55, 40, 40),
            logp=(61,),
            logscale=40,
            ring_type="conjugate_invariant",
        )
        config = CompilerConfig(margin=3, embedding_method="square")
        np = NewParameters.from_ckks_params(ckks, config)

        assert np.get_logn() == 14
        assert np.get_logq() == [55, 40, 40]
        assert np.get_logp() == [61]
        assert np.get_logscale() == 40
        assert np.get_hamming_weight() == 192
        assert np.get_ringtype() == "conjugateinvariant"
        assert np.get_margin() == 3
        assert np.get_embedding_method() == "square"
        assert np.get_max_level() == 2
        assert np.get_slots() == 2**14
        assert np.get_ring_degree() == 2**14
        assert np.get_fuse_modules() is True

    def test_from_ckks_params_standard_ring(self):
        ckks = CKKSParams(
            logn=14,
            logq=(55, 40),
            logp=(61,),
            logscale=40,
            ring_type="standard",
        )
        np = NewParameters.from_ckks_params(ckks)

        assert np.get_ringtype() == "standard"
        assert np.get_slots() == 2**13  # standard ring: logn-1

    def test_from_ckks_params_with_boot_logp(self):
        ckks = CKKSParams(
            logn=14,
            logq=(55, 40),
            logp=(61,),
            logscale=40,
            boot_logp=(50, 50),
        )
        np = NewParameters.from_ckks_params(ckks)
        assert np.get_boot_logp() == [50, 50]

    def test_from_ckks_params_default_config(self):
        ckks = CKKSParams(logn=14, logq=(55, 40), logp=(61,), logscale=40)
        np = NewParameters.from_ckks_params(ckks)
        assert np.get_margin() == 2
        assert np.get_embedding_method() == "hybrid"
        assert np.get_fuse_modules() is True

