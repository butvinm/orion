package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"

	"github.com/baahl-nyu/orion/evaluator"
)

// buildMinimalOrion constructs a minimal .orion v2 binary with a single quad node.
// Graph: input -> quad -> output. Needs RLK but no Galois keys or blobs.
func buildMinimalOrion() []byte {
	header := evaluator.CompiledHeader{
		Version: 2,
		Params: evaluator.HeaderParams{
			LogN:     12,
			LogQ:     []int{40, 30, 30},
			LogP:     []int{40},
			LogScale: 30,
			H:        192,
			RingType: "conjugate_invariant",
		},
		Config: evaluator.HeaderConfig{
			Margin:          1,
			EmbeddingMethod: "hybrid",
			FuseModules:     false,
		},
		Manifest: evaluator.HeaderManifest{
			GaloisElements: []int{},
			BootstrapSlots: []int{},
			BootLogP:       []int{},
			NeedsRLK:       true,
		},
		InputLevel: 2,
		Cost: evaluator.HeaderCost{
			BootstrapCount:    0,
			GaloisKeyCount:    0,
			BootstrapKeyCount: 0,
		},
		Graph: evaluator.HeaderGraph{
			Input:  "input",
			Output: "quad",
			Nodes: []evaluator.HeaderNode{
				{Name: "input", Op: "flatten", Level: 2, Depth: 0},
				{Name: "quad", Op: "quad", Level: 2, Depth: 1},
			},
			Edges: []evaluator.HeaderEdge{
				{Src: "input", Dst: "quad"},
			},
		},
		BlobCount: 0,
	}

	headerJSON, _ := json.Marshal(header)

	var buf bytes.Buffer
	// Magic bytes.
	buf.Write([]byte("ORION\x00\x02\x00"))
	// Header length (uint32 LE).
	headerLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(headerLen, uint32(len(headerJSON)))
	buf.Write(headerLen)
	// Header JSON.
	buf.Write(headerJSON)
	// Blob count (uint32 LE).
	blobCount := make([]byte, 4)
	binary.LittleEndian.PutUint32(blobCount, 0)
	buf.Write(blobCount)

	return buf.Bytes()
}

// testSetup creates a Server, CKKS params, and keygen context for tests.
func testSetup(t *testing.T) (*Server, ckks.Parameters, *rlwe.SecretKey) {
	t.Helper()

	data := buildMinimalOrion()
	model, err := evaluator.LoadModel(data)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	srv := NewServer(model)

	ckksParams, err := srv.CKKSParams()
	if err != nil {
		t.Fatalf("CKKSParams: %v", err)
	}

	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()

	return srv, ckksParams, sk
}

func TestHandleParams(t *testing.T) {
	srv, _, _ := testSetup(t)

	// Test GET /params returns valid JSON with expected fields.
	req := httptest.NewRequest(http.MethodGet, "/params", nil)
	w := httptest.NewRecorder()
	srv.HandleParams(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var result map[string]json.RawMessage
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	// Check required fields exist.
	for _, key := range []string{"ckks_params", "key_manifest", "input_level"} {
		if _, ok := result[key]; !ok {
			t.Errorf("missing field %q in response", key)
		}
	}

	// Verify ckks_params has logn.
	var params map[string]json.RawMessage
	if err := json.Unmarshal(result["ckks_params"], &params); err != nil {
		t.Fatalf("invalid ckks_params JSON: %v", err)
	}
	if _, ok := params["logn"]; !ok {
		t.Error("missing logn in ckks_params")
	}

	// Verify input_level.
	var inputLevel int
	if err := json.Unmarshal(result["input_level"], &inputLevel); err != nil {
		t.Fatalf("invalid input_level: %v", err)
	}
	if inputLevel != 2 {
		t.Errorf("expected input_level=2, got %d", inputLevel)
	}
}

func TestHandleParamsMethodNotAllowed(t *testing.T) {
	srv, _, _ := testSetup(t)

	req := httptest.NewRequest(http.MethodPost, "/params", nil)
	w := httptest.NewRecorder()
	srv.HandleParams(w, req)

	if w.Result().StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Result().StatusCode)
	}
}

func TestHandleSessionEmpty(t *testing.T) {
	srv, _, _ := testSetup(t)

	// Empty body should fail.
	req := httptest.NewRequest(http.MethodPost, "/session", bytes.NewReader(nil))
	w := httptest.NewRecorder()
	srv.HandleSession(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Result().StatusCode)
	}
}

func TestHandleSessionMethodNotAllowed(t *testing.T) {
	srv, _, _ := testSetup(t)

	req := httptest.NewRequest(http.MethodGet, "/session", nil)
	w := httptest.NewRecorder()
	srv.HandleSession(w, req)

	if w.Result().StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Result().StatusCode)
	}
}

func TestHandleSessionSuccess(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)

	// Generate evaluation keys (RLK only, no Galois keys needed for quad).
	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	evkBytes, err := evk.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/session", bytes.NewReader(evkBytes))
	w := httptest.NewRecorder()
	srv.HandleSession(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}

	var result sessionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if result.SessionID == "" {
		t.Error("expected non-empty session_id")
	}
}

func TestHandleInferNotFound(t *testing.T) {
	srv, _, _ := testSetup(t)

	// Use the full mux to get path parameter parsing.
	handler := srv.Handler("")
	req := httptest.NewRequest(http.MethodPost, "/session/nonexistent/infer", bytes.NewReader([]byte("data")))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Result().StatusCode)
	}
}

func TestHandleInferEmptyBody(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)

	// Create a session first.
	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evkBytes, _ := evk.MarshalBinary()

	handler := srv.Handler("")

	// Create session.
	req := httptest.NewRequest(http.MethodPost, "/session", bytes.NewReader(evkBytes))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	var sessResp sessionResponse
	json.NewDecoder(w.Result().Body).Decode(&sessResp)

	// Infer with empty body.
	req = httptest.NewRequest(http.MethodPost, "/session/"+sessResp.SessionID+"/infer", bytes.NewReader(nil))
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Result().StatusCode)
	}
}

func TestFullRoundtrip(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)

	// Generate keys.
	kg := rlwe.NewKeyGenerator(ckksParams)
	pk := kg.GenPublicKeyNew(sk)
	rlk := kg.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evkBytes, _ := evk.MarshalBinary()

	handler := srv.Handler("")

	// 1. GET /params — verify response format.
	req := httptest.NewRequest(http.MethodGet, "/params", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("GET /params: expected 200, got %d", w.Result().StatusCode)
	}

	// 2. POST /session — create session.
	req = httptest.NewRequest(http.MethodPost, "/session", bytes.NewReader(evkBytes))
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		body, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("POST /session: expected 200, got %d: %s", w.Result().StatusCode, body)
	}

	var sessResp sessionResponse
	json.NewDecoder(w.Result().Body).Decode(&sessResp)

	// 3. Encrypt input.
	encoder := ckks.NewEncoder(ckksParams)
	encryptor := ckks.NewEncryptor(ckksParams, pk)
	decryptor := ckks.NewDecryptor(ckksParams, sk)

	inputValues := make([]float64, ckksParams.MaxSlots())
	for i := range inputValues {
		inputValues[i] = 0.5
	}

	pt := ckks.NewPlaintext(ckksParams, 2) // input_level = 2
	pt.Scale = ckksParams.DefaultScale()
	encoder.Encode(inputValues, pt)
	ct, _ := encryptor.EncryptNew(pt)

	ctBytes, _ := ct.MarshalBinary()

	// 4. POST /session/{id}/infer.
	req = httptest.NewRequest(http.MethodPost, "/session/"+sessResp.SessionID+"/infer", bytes.NewReader(ctBytes))
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		body, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("POST /session/{id}/infer: expected 200, got %d: %s", w.Result().StatusCode, body)
	}

	// 5. Decrypt result.
	resultBytes, _ := io.ReadAll(w.Result().Body)
	if len(resultBytes) == 0 {
		t.Fatal("empty response body")
	}

	resultCT := &rlwe.Ciphertext{}
	if err := resultCT.UnmarshalBinary(resultBytes); err != nil {
		t.Fatalf("unmarshal result CT: %v", err)
	}

	resultPT := decryptor.DecryptNew(resultCT)
	outputValues := make([]float64, ckksParams.MaxSlots())
	encoder.Decode(resultPT, outputValues)

	// Quad = x^2. Input was 0.5, expected output ~0.25.
	expected := 0.25
	tolerance := 0.01
	for i := 0; i < 10; i++ {
		diff := outputValues[i] - expected
		if diff < -tolerance || diff > tolerance {
			t.Errorf("slot %d: expected ~%.4f, got %.4f (diff=%.6f)", i, expected, outputValues[i], diff)
		}
	}
}
