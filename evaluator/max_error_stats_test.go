package evaluator

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/stretchr/testify/require"
)

// TestMaxErrorDistribution runs multiple E2E evaluations with fresh keys
// to characterize the CKKS noise distribution. NOT for CI — this is a
// one-off calibration tool.
func TestMaxErrorDistribution(t *testing.T) {
	if os.Getenv("RUN_CALIBRATION") == "" {
		t.Skip("Skipping calibration test — set RUN_CALIBRATION=1 to run")
	}

	type testCase struct {
		name       string
		modelPath  string
		inputPath  string
		expectPath string
	}

	cases := []testCase{
		{"MLP", "testdata/mlp.orion", "testdata/mlp.input.json", "testdata/mlp.expected.json"},
		{"Sigmoid", "testdata/sigmoid.orion", "testdata/sigmoid.input.json", "testdata/sigmoid.expected.json"},
		{"SigmoidUnfused", "testdata/sigmoid_unfused.orion", "testdata/sigmoid_unfused.input.json", "testdata/sigmoid_unfused.expected.json"},
	}

	const N = 30

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := os.ReadFile(tc.modelPath)
			require.NoError(t, err)

			model, err := LoadModel(data)
			require.NoError(t, err)

			inputValues := loadJSONFloats(t, tc.inputPath)
			expectedValues := loadJSONFloats(t, tc.expectPath)

			maxSlots := model.params.MaxSlots()
			padded := make([]float64, maxSlots)
			copy(padded, inputValues)

			params, manifest, inputLevel := model.ClientParams()

			ckksParams, err := params.NewCKKSParameters()
			require.NoError(t, err)

			maxErrors := make([]float64, N)

			for run := 0; run < N; run++ {
				// Fresh keys each run to sample noise distribution.
				kg := rlwe.NewKeyGenerator(ckksParams)
				sk := kg.GenSecretKeyNew()
				pk := kg.GenPublicKeyNew(sk)
				rlk := kg.GenRelinearizationKeyNew(sk)

				galoisKeys := make([]*rlwe.GaloisKey, 0, len(manifest.GaloisElements))
				for _, ge := range manifest.GaloisElements {
					galoisKeys = append(galoisKeys, kg.GenGaloisKeyNew(ge, sk))
				}

				evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
				eval, err := NewEvaluatorFromKeySet(ckksParams, evk, nil)
				require.NoError(t, err)

				enc := ckks.NewEncoder(ckksParams)
				encryptor := ckks.NewEncryptor(ckksParams, pk)
				decryptor := ckks.NewDecryptor(ckksParams, sk)

				pt := ckks.NewPlaintext(ckksParams, inputLevel)
				pt.Scale = ckksParams.DefaultScale()
				require.NoError(t, enc.Encode(padded, pt))

				ct := ckks.NewCiphertext(ckksParams, 1, inputLevel)
				require.NoError(t, encryptor.Encrypt(pt, ct))

				results, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
				require.NoError(t, err)
				result := results[0]

				ptOut := ckks.NewPlaintext(ckksParams, result.Level())
				decryptor.Decrypt(result, ptOut)
				vals := make([]float64, maxSlots)
				require.NoError(t, enc.Decode(ptOut, vals))

				var maxErr float64
				for i, v := range expectedValues {
					d := math.Abs(v - vals[i])
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

			fmt.Printf("\n=== %s (N=%d) ===\n", tc.name, N)
			fmt.Printf("  min:    %.6f\n", min)
			fmt.Printf("  median: %.6f\n", median)
			fmt.Printf("  mean:   %.6f\n", mean)
			fmt.Printf("  p90:    %.6f\n", p90)
			fmt.Printf("  p95:    %.6f\n", p95)
			fmt.Printf("  max:    %.6f\n", max)
			fmt.Printf("  all:    %v\n", formatFloats(maxErrors))

			raw, _ := json.Marshal(maxErrors)
			fmt.Printf("  json:   %s\n", string(raw))
		})
	}
}

func formatFloats(vals []float64) string {
	s := "["
	for i, v := range vals {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%.4f", v)
	}
	return s + "]"
}
