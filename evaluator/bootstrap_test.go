package evaluator

import (
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"os"
	"sort"
	"testing"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
	"github.com/baahl-nyu/lattigo/v6/utils/bignum"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	orion "github.com/butvinm/orion"
)

// bootstrapTestParams returns CKKS params suitable for bootstrap testing at logn=14.
// These match the plan's spec: logn=14, logq=[55,40,40,40], logp=[61,61],
// log_default_scale=40, boot_logp=[61x6], ring_type=standard, h=192.
func bootstrapTestParams() orion.Params {
	return orion.Params{
		LogN:     14,
		LogQ:     []int{55, 40, 40, 40},
		LogP:     []int{61, 61},
		LogDefaultScale: 40,
		H:        192,
		RingType: "standard",
		BootLogP: []int{61, 61, 61, 61, 61, 61},
		BtpLogN:  14,
	}
}

// bootstrapTestContext holds everything needed for bootstrap tests.
type bootstrapTestContext struct {
	ckksParams ckks.Parameters
	encoder    *ckks.Encoder
	encryptor  *rlwe.Encryptor
	decryptor  *rlwe.Decryptor
	sk         *rlwe.SecretKey
	btpKeys    *bootstrapping.EvaluationKeys
	btpParams  bootstrapping.Parameters
}

// newBootstrapTestContext creates CKKS params, generates all keys (including bootstrap),
// and returns the test context. logSlots is the log2 of the number of active slots.
func newBootstrapTestContext(t *testing.T, logSlots int) *bootstrapTestContext {
	t.Helper()

	p := bootstrapTestParams()
	ckksParams, err := p.NewCKKSParameters()
	require.NoError(t, err)

	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()
	pk := kg.GenPublicKeyNew(sk)

	btpLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(14),
		LogP:     []int{61, 61, 61, 61, 61, 61},
		Xs:       ckksParams.Xs(),
		LogSlots: utils.Pointy(logSlots),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
	require.NoError(t, err)

	btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
	require.NoError(t, err)

	return &bootstrapTestContext{
		ckksParams: ckksParams,
		encoder:    ckks.NewEncoder(ckksParams),
		encryptor:  ckks.NewEncryptor(ckksParams, pk),
		decryptor:  ckks.NewDecryptor(ckksParams, sk),
		sk:         sk,
		btpKeys:    btpKeys,
		btpParams:  btpParams,
	}
}

// buildSyntheticBootstrapModel creates a synthetic Model with a flatten -> bootstrap graph.
func buildSyntheticBootstrapModel(t *testing.T, ckksParams ckks.Parameters, p orion.Params, cfg BootstrapConfig, inputLevel int) *Model {
	t.Helper()

	cfgJSON, err := json.Marshal(cfg)
	require.NoError(t, err)

	header := &CompiledHeader{
		Version: 2,
		Params: HeaderParams{
			LogN:     p.LogN,
			LogQ:     p.LogQ,
			LogP:     p.LogP,
			LogDefaultScale: p.LogDefaultScale,
			H:        p.H,
			RingType: p.RingType,
			BootLogP: p.BootLogP,
			BtpLogN:  p.BtpLogN,
		},
		Config: HeaderConfig{
			Margin:          1,
			EmbeddingMethod: "hybrid",
			FuseModules:     true,
		},
		Manifest: HeaderManifest{
			GaloisElements: []int{},
			BootstrapSlots: []int{cfg.Slots},
			BootLogP:       p.BootLogP,
			BtpLogN:        p.BtpLogN,
			NeedsRLK:       false,
		},
		InputLevel: inputLevel,
		Graph: HeaderGraph{
			Input:  "flatten",
			Output: "bootstrap_0",
			Nodes: []HeaderNode{
				{Name: "flatten", Op: "flatten", Level: inputLevel},
				{Name: "bootstrap_0", Op: "bootstrap", Level: inputLevel, Config: cfgJSON},
			},
			Edges: []HeaderEdge{
				{Src: "flatten", Dst: "bootstrap_0"},
			},
		},
		BlobCount: 0,
	}

	graph, err := buildGraph(header)
	require.NoError(t, err)

	return &Model{
		header:      header,
		clientParam: p,
		params:      ckksParams,
		graph:       graph,
		rawBlobs:    nil,
		biases:      make(map[string]*rlwe.Plaintext),
		polys:       make(map[string]bignum.Polynomial),
		ltConfigs:   make(map[string]*LinearTransformConfig),
		polyConfigs: make(map[string]*PolynomialConfig),
	}
}

// TestEvalBootstrap tests the bootstrap operation at multiple input levels.
// The compiler guarantees input_level >= 1 (level_dag.py:238-240). The prescale
// multiply always runs (even when prescale==1) to zero inactive slots, consuming
// 1 level. We test at levels 1, 2, and 3 to cover minimum, middle, and high.
func TestEvalBootstrap(t *testing.T) {
	logSlots := 7 // 128 slots
	slots := 1 << logSlots

	btpCtx := newBootstrapTestContext(t, logSlots)
	p := bootstrapTestParams()
	// logq=[55,40,40,40] → max level 3

	kg := rlwe.NewKeyGenerator(btpCtx.ckksParams)
	rlk := kg.GenRelinearizationKeyNew(btpCtx.sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	for _, inputLevel := range []int{1, 2} {
		t.Run(fmt.Sprintf("level_%d", inputLevel), func(t *testing.T) {
			eval, err := NewEvaluatorFromKeySet(btpCtx.ckksParams, evk, btpCtx.btpKeys)
			require.NoError(t, err)
			defer eval.Close()

			cfg := BootstrapConfig{
				InputLevel: inputLevel,
				InputMin:   -1.0,
				InputMax:   1.0,
				Prescale:   1.0,
				Postscale:  1.0,
				Constant:   0.0,
				Slots:      slots,
			}
			model := buildSyntheticBootstrapModel(t, btpCtx.ckksParams, p, cfg, inputLevel)

			maxSlots := btpCtx.ckksParams.MaxSlots()
			input := make([]float64, maxSlots)
			for i := 0; i < slots; i++ {
				input[i] = 2.0*float64(i)/float64(slots) - 1.0
			}

			ctx := &testContext{
				ckksParams: btpCtx.ckksParams,
				encoder:    btpCtx.encoder,
				encryptor:  btpCtx.encryptor,
				decryptor:  btpCtx.decryptor,
			}
			ct := encryptVector(t, ctx, input, inputLevel)
			require.Equal(t, inputLevel, ct.Level())

			result, err := eval.Forward(model, ct)
			require.NoError(t, err)

			assert.Greater(t, result.Level(), inputLevel,
				"bootstrap should refresh level: got %d, want > %d", result.Level(), inputLevel)

			decoded := decryptVector(t, ctx, result)

			tolerance := 0.01
			var maxErr float64
			for i := 0; i < slots; i++ {
				diff := math.Abs(input[i] - decoded[i])
				if diff > maxErr {
					maxErr = diff
				}
			}
			t.Logf("Bootstrap (logSlots=%d, level %d->%d): max_error=%.6f tolerance=%.6f",
				logSlots, inputLevel, result.Level(), maxErr, tolerance)

			for i := 0; i < slots; i++ {
				assert.InDelta(t, input[i], decoded[i], tolerance,
					"slot %d: expected %.6f, got %.6f", i, input[i], decoded[i])
			}
		})
	}
}

// TestEvalBootstrapRejectsLevel0 verifies that bootstrap at level 0 returns a
// clear error. The compiler never produces level 0 bootstrap nodes, but the
// evaluator should fail explicitly rather than with an opaque Lattigo error.
func TestEvalBootstrapRejectsLevel0(t *testing.T) {
	logSlots := 7
	slots := 1 << logSlots

	btpCtx := newBootstrapTestContext(t, logSlots)
	p := bootstrapTestParams()

	kg := rlwe.NewKeyGenerator(btpCtx.ckksParams)
	rlk := kg.GenRelinearizationKeyNew(btpCtx.sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval, err := NewEvaluatorFromKeySet(btpCtx.ckksParams, evk, btpCtx.btpKeys)
	require.NoError(t, err)
	defer eval.Close()

	inputLevel := 0
	cfg := BootstrapConfig{
		InputLevel: inputLevel,
		InputMin:   -1.0,
		InputMax:   1.0,
		Prescale:   1.0,
		Postscale:  1.0,
		Constant:   0.0,
		Slots:      slots,
	}
	model := buildSyntheticBootstrapModel(t, btpCtx.ckksParams, p, cfg, inputLevel)

	maxSlots := btpCtx.ckksParams.MaxSlots()
	input := make([]float64, maxSlots)
	for i := 0; i < slots; i++ {
		input[i] = 2.0*float64(i)/float64(slots) - 1.0
	}

	ctx := &testContext{
		ckksParams: btpCtx.ckksParams,
		encoder:    btpCtx.encoder,
		encryptor:  btpCtx.encryptor,
		decryptor:  btpCtx.decryptor,
	}
	ct := encryptVector(t, ctx, input, inputLevel)

	_, err = eval.Forward(model, ct)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "input level >= 1")
}

// TestForwardWithBootstrap runs a full E2E test with a compiled model that includes
// bootstrap. Uses the bootstrap MLP fixture generated by generate.py.
func TestForwardWithBootstrap(t *testing.T) {
	modelPath := "testdata/bootstrap_mlp.orion"
	inputPath := "testdata/bootstrap_mlp.input.json"
	expectedPath := "testdata/bootstrap_mlp.expected.json"

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("bootstrap_mlp fixture not found — run 'python evaluator/testdata/generate.py' to generate")
	}

	data, err := os.ReadFile(modelPath)
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, inputLevel := model.ClientParams()
	ckksParams, err := params.NewCKKSParameters()
	require.NoError(t, err)

	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()
	pk := kg.GenPublicKeyNew(sk)

	var rlk *rlwe.RelinearizationKey
	if manifest.NeedsRLK {
		rlk = kg.GenRelinearizationKeyNew(sk)
	}

	galoisKeys := make([]*rlwe.GaloisKey, 0, len(manifest.GaloisElements))
	for _, ge := range manifest.GaloisElements {
		galoisKeys = append(galoisKeys, kg.GenGaloisKeyNew(ge, sk))
	}
	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)

	var btpKeys *bootstrapping.EvaluationKeys
	if len(manifest.BootstrapSlots) > 0 {
		logSlots := bits.Len(uint(manifest.BootstrapSlots[0])) - 1
		btpLit := bootstrapping.ParametersLiteral{
			LogN:     utils.Pointy(manifest.BtpLogN),
			LogP:     manifest.BootLogP,
			Xs:       ckksParams.Xs(),
			LogSlots: utils.Pointy(logSlots),
		}
		btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
		require.NoError(t, err)

		btpKeys, _, err = btpParams.GenEvaluationKeys(sk)
		require.NoError(t, err)
	}

	eval, err := NewEvaluatorFromKeySet(ckksParams, evk, btpKeys)
	require.NoError(t, err)
	defer eval.Close()

	inputValues := loadJSONFloats(t, inputPath)
	maxSlots := ckksParams.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, inputValues)

	ctx := &testContext{
		ckksParams: ckksParams,
		encoder:    ckks.NewEncoder(ckksParams),
		encryptor:  ckks.NewEncryptor(ckksParams, pk),
		decryptor:  ckks.NewDecryptor(ckksParams, sk),
	}
	ct := encryptVector(t, ctx, padded, inputLevel)

	result, err := eval.Forward(model, ct)
	require.NoError(t, err, "Forward failed")

	decoded := decryptVector(t, ctx, result)
	expected := loadJSONFloats(t, expectedPath)

	// Bootstrap adds noise on top of regular CKKS noise.
	tolerance := 0.5

	var maxErr float64
	for i, v := range expected {
		diff := math.Abs(v - decoded[i])
		if diff > maxErr {
			maxErr = diff
		}
		t.Logf("  [%d] expected=%.6f got=%.6f diff=%.6f", i, v, decoded[i], diff)
	}
	t.Logf("Max error: %.6f", maxErr)

	for i, v := range expected {
		assert.InDelta(t, v, decoded[i], tolerance,
			"slot %d: expected %f, got %f", i, v, decoded[i])
	}
}

// TestBootstrapMaxErrorDistribution runs bootstrap tolerance calibration.
// NOT for CI — use for one-off calibration.
func TestBootstrapMaxErrorDistribution(t *testing.T) {
	if os.Getenv("RUN_CALIBRATION") == "" {
		t.Skip("Skipping calibration test — set RUN_CALIBRATION=1 to run")
	}

	const N = 30
	logSlots := 7
	slots := 1 << logSlots

	p := bootstrapTestParams()
	ckksParams, err := p.NewCKKSParameters()
	require.NoError(t, err)

	inputLevel := 1 // compiler guarantees bootstrap input_level >= 1

	maxSlots := ckksParams.MaxSlots()
	input := make([]float64, maxSlots)
	for i := 0; i < slots; i++ {
		input[i] = 2.0*float64(i)/float64(slots) - 1.0
	}

	cfg := BootstrapConfig{
		InputLevel: inputLevel,
		InputMin:   -1.0,
		InputMax:   1.0,
		Prescale:   1.0,
		Postscale:  1.0,
		Constant:   0.0,
		Slots:      slots,
	}

	maxErrors := make([]float64, N)

	for run := 0; run < N; run++ {
		kg := rlwe.NewKeyGenerator(ckksParams)
		sk := kg.GenSecretKeyNew()
		pk := kg.GenPublicKeyNew(sk)
		rlk := kg.GenRelinearizationKeyNew(sk)

		btpLit := bootstrapping.ParametersLiteral{
			LogN:     utils.Pointy(14),
			LogP:     []int{61, 61, 61, 61, 61, 61},
			Xs:       ckksParams.Xs(),
			LogSlots: utils.Pointy(logSlots),
		}
		btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
		require.NoError(t, err)

		btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
		require.NoError(t, err)

		evk := rlwe.NewMemEvaluationKeySet(rlk)
		eval, err := NewEvaluatorFromKeySet(ckksParams, evk, btpKeys)
		require.NoError(t, err)

		model := buildSyntheticBootstrapModel(t, ckksParams, p, cfg, inputLevel)

		enc := ckks.NewEncoder(ckksParams)
		encryptor := ckks.NewEncryptor(ckksParams, pk)
		decryptor := ckks.NewDecryptor(ckksParams, sk)

		pt := ckks.NewPlaintext(ckksParams, inputLevel)
		pt.Scale = ckksParams.DefaultScale()
		require.NoError(t, enc.Encode(input, pt))

		ct := ckks.NewCiphertext(ckksParams, 1, inputLevel)
		require.NoError(t, encryptor.Encrypt(pt, ct))

		result, err := eval.Forward(model, ct)
		require.NoError(t, err)

		ptOut := ckks.NewPlaintext(ckksParams, result.Level())
		decryptor.Decrypt(result, ptOut)
		vals := make([]float64, maxSlots)
		require.NoError(t, enc.Decode(ptOut, vals))

		var maxErr float64
		for i := 0; i < slots; i++ {
			d := math.Abs(input[i] - vals[i])
			if d > maxErr {
				maxErr = d
			}
		}
		maxErrors[run] = maxErr

		eval.Close()
	}

	sort.Float64s(maxErrors)
	min := maxErrors[0]
	max := maxErrors[N-1]
	median := maxErrors[N/2]
	p90 := maxErrors[N-N/10-1]
	p95 := maxErrors[N-N/20-1]

	var sum float64
	for _, e := range maxErrors {
		sum += e
	}
	mean := sum / float64(N)

	fmt.Printf("\n=== Bootstrap (N=%d, logSlots=%d) ===\n", N, logSlots)
	fmt.Printf("  min:    %.6f\n", min)
	fmt.Printf("  median: %.6f\n", median)
	fmt.Printf("  mean:   %.6f\n", mean)
	fmt.Printf("  p90:    %.6f\n", p90)
	fmt.Printf("  p95:    %.6f\n", p95)
	fmt.Printf("  max:    %.6f\n", max)
	fmt.Printf("  recommended tolerance (max*1.5): %.6f\n", max*1.5)

	raw, _ := json.Marshal(maxErrors)
	fmt.Printf("  json:   %s\n", string(raw))
}
