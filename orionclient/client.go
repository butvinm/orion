package orionclient

import (
	"fmt"
	"math"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// Client holds the secret key and performs encode/decode, encrypt/decrypt,
// and key generation. Multiple Client instances can coexist.
type Client struct {
	params     Params
	ckksParams ckks.Parameters
	keygen     *rlwe.KeyGenerator
	sk         *rlwe.SecretKey
	pk         *rlwe.PublicKey
	encoder    *ckks.Encoder
	encryptor  *rlwe.Encryptor
	decryptor  *rlwe.Decryptor
}

// New creates a Client with freshly generated secret and public keys.
func New(p Params) (*Client, error) {
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}

	keygen := rlwe.NewKeyGenerator(ckksParams)
	sk := keygen.GenSecretKeyNew()
	pk := keygen.GenPublicKeyNew(sk)

	enc := ckks.NewEncoder(ckksParams)
	encryptor := ckks.NewEncryptor(ckksParams, pk)
	decryptor := ckks.NewDecryptor(ckksParams, sk)

	return &Client{
		params:     p,
		ckksParams: ckksParams,
		keygen:     keygen,
		sk:         sk,
		pk:         pk,
		encoder:    enc,
		encryptor:  encryptor,
		decryptor:  decryptor,
	}, nil
}

// FromSecretKey restores a Client from a serialized secret key.
func FromSecretKey(p Params, skData []byte) (*Client, error) {
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}

	sk := &rlwe.SecretKey{}
	if err := sk.UnmarshalBinary(skData); err != nil {
		return nil, fmt.Errorf("unmarshalling secret key: %w", err)
	}

	keygen := rlwe.NewKeyGenerator(ckksParams)
	pk := keygen.GenPublicKeyNew(sk)

	enc := ckks.NewEncoder(ckksParams)
	encryptor := ckks.NewEncryptor(ckksParams, pk)
	decryptor := ckks.NewDecryptor(ckksParams, sk)

	return &Client{
		params:     p,
		ckksParams: ckksParams,
		keygen:     keygen,
		sk:         sk,
		pk:         pk,
		encoder:    enc,
		encryptor:  encryptor,
		decryptor:  decryptor,
	}, nil
}

// Close zeros the secret key memory and releases resources.
func (c *Client) Close() {
	if c.sk != nil {
		// Zero the secret key polynomial coefficients
		for i := range c.sk.Value.Q.Coeffs {
			for j := range c.sk.Value.Q.Coeffs[i] {
				c.sk.Value.Q.Coeffs[i][j] = 0
			}
		}
		for i := range c.sk.Value.P.Coeffs {
			for j := range c.sk.Value.P.Coeffs[i] {
				c.sk.Value.P.Coeffs[i][j] = 0
			}
		}
		c.sk = nil
	}
	c.keygen = nil
	c.pk = nil
	c.encryptor = nil
	c.decryptor = nil
	c.encoder = nil
}

// SecretKey serializes the secret key for persistence.
func (c *Client) SecretKey() ([]byte, error) {
	if c.sk == nil {
		return nil, fmt.Errorf("client is closed or secret key is nil")
	}
	return c.sk.MarshalBinary()
}

// Encode encodes a float64 slice into a Plaintext at the given level and scale.
// The values slice length must not exceed MaxSlots().
func (c *Client) Encode(values []float64, level int, scale uint64) (*Plaintext, error) {
	if c.encoder == nil {
		return nil, fmt.Errorf("client is closed")
	}

	pt := ckks.NewPlaintext(c.ckksParams, level)
	pt.Scale = rlwe.NewScale(scale)

	if err := c.encoder.Encode(values, pt); err != nil {
		return nil, fmt.Errorf("encoding: %w", err)
	}

	return &Plaintext{
		raw:   pt,
		shape: []int{len(values)},
	}, nil
}

// Decode decodes a Plaintext back to float64 values.
func (c *Client) Decode(pt *Plaintext) ([]float64, error) {
	if c.encoder == nil {
		return nil, fmt.Errorf("client is closed")
	}

	result := make([]float64, c.ckksParams.MaxSlots())
	if err := c.encoder.Decode(pt.raw, result); err != nil {
		return nil, fmt.Errorf("decoding: %w", err)
	}

	return result, nil
}

// Encrypt encrypts a Plaintext into a Ciphertext (with one underlying ct).
func (c *Client) Encrypt(pt *Plaintext) (*Ciphertext, error) {
	if c.encryptor == nil {
		return nil, fmt.Errorf("client is closed")
	}

	ct := ckks.NewCiphertext(c.ckksParams, 1, pt.raw.Level())
	if err := c.encryptor.Encrypt(pt.raw, ct); err != nil {
		return nil, fmt.Errorf("encrypting: %w", err)
	}

	shape := make([]int, len(pt.shape))
	copy(shape, pt.shape)

	return &Ciphertext{
		cts:   []*rlwe.Ciphertext{ct},
		shape: shape,
	}, nil
}

// Decrypt decrypts all ciphertexts in a Ciphertext, returning one Plaintext per ct.
func (c *Client) Decrypt(ct *Ciphertext) ([]*Plaintext, error) {
	if c.decryptor == nil {
		return nil, fmt.Errorf("client is closed")
	}

	pts := make([]*Plaintext, len(ct.cts))
	for i, ciphertext := range ct.cts {
		pt := ckks.NewPlaintext(c.ckksParams, ciphertext.Level())
		c.decryptor.Decrypt(ciphertext, pt)
		pts[i] = &Plaintext{
			raw:   pt,
			shape: ct.Shape(),
		}
	}

	return pts, nil
}

// GenerateRLK generates and serializes a relinearization key.
func (c *Client) GenerateRLK() ([]byte, error) {
	if c.keygen == nil || c.sk == nil {
		return nil, fmt.Errorf("client is closed")
	}

	rlk := c.keygen.GenRelinearizationKeyNew(c.sk)
	return rlk.MarshalBinary()
}

// GenerateGaloisKey generates and serializes a Galois key for the given element.
func (c *Client) GenerateGaloisKey(galoisElement uint64) ([]byte, error) {
	if c.keygen == nil || c.sk == nil {
		return nil, fmt.Errorf("client is closed")
	}

	gk := c.keygen.GenGaloisKeyNew(galoisElement, c.sk)
	return gk.MarshalBinary()
}

// GenerateBootstrapKeys generates and serializes bootstrap evaluation keys
// for the given slot count and auxiliary primes.
func (c *Client) GenerateBootstrapKeys(slots int, logP []int) ([]byte, error) {
	if c.keygen == nil || c.sk == nil {
		return nil, fmt.Errorf("client is closed")
	}

	btpLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(c.ckksParams.LogN()),
		LogP:     logP,
		Xs:       c.ckksParams.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(slots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(c.ckksParams, btpLit)
	if err != nil {
		return nil, fmt.Errorf("creating bootstrap parameters: %w", err)
	}

	btpKeys, _, err := btpParams.GenEvaluationKeys(c.sk)
	if err != nil {
		return nil, fmt.Errorf("generating bootstrap keys: %w", err)
	}

	return btpKeys.MarshalBinary()
}

// GenerateKeys generates all evaluation keys specified by a Manifest.
func (c *Client) GenerateKeys(manifest Manifest) (*EvalKeyBundle, error) {
	bundle := &EvalKeyBundle{
		GaloisKeys:    make(map[uint64][]byte),
		BootstrapKeys: make(map[int][]byte),
		BootLogP:      manifest.BootLogP,
	}

	if manifest.NeedsRLK {
		rlk, err := c.GenerateRLK()
		if err != nil {
			return nil, fmt.Errorf("generating RLK: %w", err)
		}
		bundle.RLK = rlk
	}

	for _, galEl := range manifest.GaloisElements {
		gk, err := c.GenerateGaloisKey(galEl)
		if err != nil {
			return nil, fmt.Errorf("generating Galois key for element %d: %w", galEl, err)
		}
		bundle.GaloisKeys[galEl] = gk
	}

	if len(manifest.BootstrapSlots) > 0 && len(manifest.BootLogP) > 0 {
		for _, slots := range manifest.BootstrapSlots {
			bk, err := c.GenerateBootstrapKeys(slots, manifest.BootLogP)
			if err != nil {
				return nil, fmt.Errorf("generating bootstrap keys for %d slots: %w", slots, err)
			}
			bundle.BootstrapKeys[slots] = bk
		}
	}

	return bundle, nil
}

// MaxSlots returns the maximum number of plaintext slots.
func (c *Client) MaxSlots() int {
	return c.ckksParams.MaxSlots()
}

// DefaultScale returns the default scale (2^LogScale).
func (c *Client) DefaultScale() uint64 {
	return c.params.DefaultScale()
}

// GaloisElement returns the Galois element for a given rotation step.
func (c *Client) GaloisElement(rotation int) uint64 {
	return c.ckksParams.GaloisElement(rotation)
}

// ModuliChain returns the actual Q moduli (NTT-friendly primes) used by the scheme.
func (c *Client) ModuliChain() []uint64 {
	return c.ckksParams.Q()
}

// AuxModuliChain returns the P (auxiliary) moduli used by the scheme.
func (c *Client) AuxModuliChain() []uint64 {
	return c.ckksParams.P()
}
