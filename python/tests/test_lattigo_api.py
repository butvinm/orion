"""Tests for the lattigo Python package — Lattigo primitive API.

Tests keygen, encode/decode roundtrip, encrypt/decrypt roundtrip,
and MemEvaluationKeySet marshal/unmarshal using the new Lattigo-primitive API.
"""

import pytest
from lattigo.ckks import Encoder, Parameters
from lattigo.rlwe import (
    Ciphertext,
    Decryptor,
    Encryptor,
    GaloisKey,
    KeyGenerator,
    MemEvaluationKeySet,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    SecretKey,
)

# Small CKKS parameters for fast tests
TEST_LOGN = 13
TEST_LOGQ = [55, 40, 40, 40, 40, 40]
TEST_LOGP = [56]
TEST_LOGSCALE = 40
TEST_NUM_SLOTS = 1 << TEST_LOGN  # conjugate_invariant → 2^logn slots


@pytest.fixture
def params():
    p = Parameters(
        logn=TEST_LOGN,
        logq=TEST_LOGQ,
        logp=TEST_LOGP,
        log_default_scale=TEST_LOGSCALE,
        ring_type="conjugate_invariant",
    )
    yield p
    p.close()


@pytest.fixture
def keygen(params):
    kg = KeyGenerator(params)
    yield kg
    kg.close()


@pytest.fixture
def sk(keygen):
    key = keygen.gen_secret_key()
    yield key
    key.close()


@pytest.fixture
def pk(keygen, sk):
    key = keygen.gen_public_key(sk)
    yield key
    key.close()


class TestParameters:
    def test_max_slots(self, params):
        assert params.max_slots() == TEST_NUM_SLOTS

    def test_max_level(self, params):
        assert params.max_level() == len(TEST_LOGQ) - 1

    def test_default_scale(self, params):
        assert params.default_scale() == 1 << TEST_LOGSCALE

    def test_galois_element(self, params):
        # Galois element for rotation=1 should be a non-zero uint64
        gel = params.galois_element(1)
        assert gel > 0


class TestKeyGeneration:
    def test_gen_secret_key(self, keygen):
        sk = keygen.gen_secret_key()
        assert sk._handle
        sk.close()

    def test_gen_public_key(self, keygen, sk):
        pk = keygen.gen_public_key(sk)
        assert pk._handle
        pk.close()

    def test_gen_relin_key(self, keygen, sk):
        rlk = keygen.gen_relin_key(sk)
        assert rlk._handle
        rlk.close()

    def test_gen_galois_key(self, keygen, sk, params):
        gel = params.galois_element(1)
        gk = keygen.gen_galois_key(sk, gel)
        assert gk._handle
        gk.close()


class TestSecretKeySerialization:
    def test_marshal_unmarshal_roundtrip(self, sk):
        data = sk.marshal_binary()
        assert len(data) > 0

        sk2 = SecretKey.unmarshal_binary(data)
        data2 = sk2.marshal_binary()
        assert data == data2
        sk2.close()


class TestPublicKeySerialization:
    def test_marshal_unmarshal_roundtrip(self, pk):
        data = pk.marshal_binary()
        assert len(data) > 0

        pk2 = PublicKey.unmarshal_binary(data)
        data2 = pk2.marshal_binary()
        assert data == data2
        pk2.close()


class TestRelinearizationKeySerialization:
    def test_marshal_unmarshal_roundtrip(self, keygen, sk):
        rlk = keygen.gen_relin_key(sk)
        data = rlk.marshal_binary()
        assert len(data) > 0

        rlk2 = RelinearizationKey.unmarshal_binary(data)
        data2 = rlk2.marshal_binary()
        assert data == data2
        rlk2.close()
        rlk.close()


class TestGaloisKeySerialization:
    def test_marshal_unmarshal_roundtrip(self, keygen, sk, params):
        gel = params.galois_element(1)
        gk = keygen.gen_galois_key(sk, gel)
        data = gk.marshal_binary()
        assert len(data) > 0

        gk2 = GaloisKey.unmarshal_binary(data)
        data2 = gk2.marshal_binary()
        assert data == data2
        gk2.close()
        gk.close()


class TestEncodeDecode:
    def test_encode_decode_roundtrip(self, params, sk):
        encoder = Encoder(params)
        level = params.max_level()
        scale = params.default_scale()

        values = [float(i) for i in range(64)]
        pt = encoder.encode(values, level, scale)
        assert isinstance(pt, Plaintext)
        assert pt.level() == level

        decoded = encoder.decode(pt, params.max_slots())
        # CKKS is approximate — check within tolerance
        for i in range(len(values)):
            assert abs(decoded[i] - values[i]) < 1e-7, (
                f"decoded[{i}]={decoded[i]} != {values[i]}"
            )

        pt.close()
        encoder.close()


class TestEncryptDecrypt:
    def test_encrypt_decrypt_roundtrip(self, params, keygen, sk, pk):
        encoder = Encoder(params)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        level = params.max_level()
        scale = params.default_scale()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Encode → Encrypt → Decrypt → Decode
        pt = encoder.encode(values, level, scale)
        ct = encryptor.encrypt_new(pt)
        assert isinstance(ct, Ciphertext)
        assert ct.level() == level

        pt_dec = decryptor.decrypt_new(ct)
        decoded = encoder.decode(pt_dec, params.max_slots())

        for i in range(len(values)):
            assert abs(decoded[i] - values[i]) < 1e-4, (
                f"decoded[{i}]={decoded[i]} != {values[i]}"
            )

        pt.close()
        ct.close()
        pt_dec.close()
        encryptor.close()
        decryptor.close()
        encoder.close()


class TestCiphertextSerialization:
    def test_marshal_unmarshal_roundtrip(self, params, pk, sk):
        encoder = Encoder(params)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        level = params.max_level()
        scale = params.default_scale()
        values = [10.0, 20.0, 30.0]

        pt = encoder.encode(values, level, scale)
        ct = encryptor.encrypt_new(pt)

        # Marshal / unmarshal the ciphertext
        data = ct.marshal_binary()
        assert len(data) > 0

        ct2 = Ciphertext.unmarshal_binary(data)
        assert ct2.level() == ct.level()

        # Decrypt the deserialized ciphertext
        pt_dec = decryptor.decrypt_new(ct2)
        decoded = encoder.decode(pt_dec, params.max_slots())

        for i in range(len(values)):
            assert abs(decoded[i] - values[i]) < 1e-4, (
                f"decoded[{i}]={decoded[i]} != {values[i]}"
            )

        pt.close()
        ct.close()
        ct2.close()
        pt_dec.close()
        encryptor.close()
        decryptor.close()
        encoder.close()


class TestPlaintextSerialization:
    def test_marshal_unmarshal_roundtrip(self, params):
        encoder = Encoder(params)
        level = params.max_level()
        scale = params.default_scale()

        values = [7.0, 8.0, 9.0]
        pt = encoder.encode(values, level, scale)

        data = pt.marshal_binary()
        assert len(data) > 0

        pt2 = Plaintext.unmarshal_binary(data)
        assert pt2.level() == pt.level()

        decoded = encoder.decode(pt2, params.max_slots())
        for i in range(len(values)):
            assert abs(decoded[i] - values[i]) < 1e-7

        pt.close()
        pt2.close()
        encoder.close()


class TestMemEvaluationKeySet:
    def test_create_with_rlk_and_galois_keys(self, params, keygen, sk):
        rlk = keygen.gen_relin_key(sk)
        gel = params.galois_element(1)
        gk = keygen.gen_galois_key(sk, gel)

        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=[gk])
        assert evk._handle
        evk.close()
        rlk.close()
        gk.close()

    def test_create_with_rlk_only(self, keygen, sk):
        rlk = keygen.gen_relin_key(sk)
        evk = MemEvaluationKeySet(rlk=rlk)
        assert evk._handle
        evk.close()
        rlk.close()

    def test_marshal_unmarshal_roundtrip(self, params, keygen, sk):
        rlk = keygen.gen_relin_key(sk)
        gel1 = params.galois_element(1)
        gel2 = params.galois_element(-1)
        gk1 = keygen.gen_galois_key(sk, gel1)
        gk2 = keygen.gen_galois_key(sk, gel2)

        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=[gk1, gk2])

        # Marshal
        data = evk.marshal_binary()
        assert len(data) > 0

        # Unmarshal
        evk2 = MemEvaluationKeySet.unmarshal_binary(data)
        assert evk2._handle

        # Re-marshal and compare
        data2 = evk2.marshal_binary()
        assert data == data2

        evk.close()
        evk2.close()
        rlk.close()
        gk1.close()
        gk2.close()
