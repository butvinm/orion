package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
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

// buildMinimalOrionWithGalois builds an .orion v2 binary that requires specific Galois elements.
func buildMinimalOrionWithGalois(galoisElements []int) []byte {
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
			GaloisElements: galoisElements,
			BootstrapSlots: []int{},
			BootLogP:       []int{},
			NeedsRLK:       true,
		},
		InputLevel: 2,
		Cost: evaluator.HeaderCost{
			BootstrapCount:    0,
			GaloisKeyCount:    len(galoisElements),
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
	buf.Write([]byte("ORION\x00\x02\x00"))
	headerLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(headerLen, uint32(len(headerJSON)))
	buf.Write(headerLen)
	buf.Write(headerJSON)
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

	srv, err := NewServer(model)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	ckksParams := srv.CKKSParams()
	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()

	return srv, ckksParams, sk
}

// testSetupWithGalois creates a Server with a model requiring specific Galois elements.
func testSetupWithGalois(t *testing.T, galoisElements []int) (*Server, ckks.Parameters, *rlwe.SecretKey) {
	t.Helper()

	data := buildMinimalOrionWithGalois(galoisElements)
	model, err := evaluator.LoadModel(data)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	srv, err := NewServer(model)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	ckksParams := srv.CKKSParams()
	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()

	return srv, ckksParams, sk
}

// createSession is a test helper that creates a pending session and returns its ID.
func createSession(t *testing.T, handler http.Handler) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/session", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("POST /session: expected 200, got %d: %s", resp.StatusCode, body)
	}

	var result sessionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("invalid session response JSON: %v", err)
	}
	return result.SessionID
}

// uploadRLK is a test helper that uploads a relinearization key to a session.
func uploadRLK(t *testing.T, handler http.Handler, sessionID string, rlkBytes []byte) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/session/"+sessionID+"/keys/relin", bytes.NewReader(rlkBytes))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		body, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("POST /keys/relin: expected 200, got %d: %s", w.Result().StatusCode, body)
	}
}

// uploadGaloisKey is a test helper that uploads a Galois key to a session.
func uploadGaloisKey(t *testing.T, handler http.Handler, sessionID string, element uint64, gkBytes []byte) {
	t.Helper()
	url := fmt.Sprintf("/session/%s/keys/galois/%d", sessionID, element)
	req := httptest.NewRequest(http.MethodPost, url, bytes.NewReader(gkBytes))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		body, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("POST /keys/galois/%d: expected 200, got %d: %s", element, w.Result().StatusCode, body)
	}
}

// finalizeSession is a test helper that finalizes a session.
func finalizeSession(t *testing.T, handler http.Handler, sessionID string) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/session/"+sessionID+"/keys/finalize", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		body, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("POST /keys/finalize: expected 200, got %d: %s", w.Result().StatusCode, body)
	}
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

func TestHandleSessionCreatesPendingSession(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := srv.Handler("")

	id := createSession(t, handler)
	if len(id) != 32 {
		t.Errorf("expected session_id length 32, got %d: %q", len(id), id)
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

func TestHandleRelinKey(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, err := rlk.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
}

func TestHandleRelinKeyRejectReady(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, err := rlk.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
	finalizeSession(t, handler, id)

	// Upload to ready session should get 409.
	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/keys/relin", bytes.NewReader(rlkBytes))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusConflict {
		t.Errorf("expected 409 for ready session, got %d", w.Result().StatusCode)
	}
}

func TestHandleRelinKeyEmpty(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := srv.Handler("")

	id := createSession(t, handler)

	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/keys/relin", bytes.NewReader(nil))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400 for empty body, got %d", w.Result().StatusCode)
	}
}

func TestHandleGaloisKey(t *testing.T) {
	// Get a valid Galois element for the ring.
	tmpSrv, tmpParams, _ := testSetup(t)
	_ = tmpSrv
	ge := tmpParams.GaloisElement(1)

	srv, ckksParams, sk := testSetupWithGalois(t, []int{int(ge)})
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	gk := kg.GenGaloisKeyNew(ge, sk)
	gkBytes, err := gk.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	id := createSession(t, handler)
	uploadGaloisKey(t, handler, id, ge, gkBytes)
}

func TestHandleGaloisKeyIdempotent(t *testing.T) {
	tmpSrv, tmpParams, _ := testSetup(t)
	_ = tmpSrv
	ge := tmpParams.GaloisElement(1)

	srv, ckksParams, sk := testSetupWithGalois(t, []int{int(ge)})
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	gk := kg.GenGaloisKeyNew(ge, sk)
	gkBytes, err := gk.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	id := createSession(t, handler)
	// Upload same key twice — should succeed both times.
	uploadGaloisKey(t, handler, id, ge, gkBytes)
	uploadGaloisKey(t, handler, id, ge, gkBytes)
}

func TestHandleGaloisKeyRejectReady(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)
	handler := srv.Handler("")

	ge := ckksParams.GaloisElement(1)
	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, _ := rlk.MarshalBinary()

	gk := kg.GenGaloisKeyNew(ge, sk)
	gkBytes, _ := gk.MarshalBinary()

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
	finalizeSession(t, handler, id)

	req := httptest.NewRequest(http.MethodPost, fmt.Sprintf("/session/%s/keys/galois/%d", id, ge), bytes.NewReader(gkBytes))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusConflict {
		t.Errorf("expected 409 for ready session, got %d", w.Result().StatusCode)
	}
}

func TestHandleGaloisKeyInvalidElement(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := srv.Handler("")

	id := createSession(t, handler)

	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/keys/galois/notanumber", bytes.NewReader([]byte("data")))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid element, got %d", w.Result().StatusCode)
	}
}

func TestHandleFinalizeSuccess(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, _ := rlk.MarshalBinary()

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
	finalizeSession(t, handler, id)
}

func TestHandleFinalizeMissingRLK(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := srv.Handler("")

	id := createSession(t, handler)

	// Finalize without uploading RLK — model needs RLK.
	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/keys/finalize", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Result().StatusCode)
	}

	var errResp finalizeErrorResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&errResp); err != nil {
		t.Fatalf("decoding error response: %v", err)
	}
	if !errResp.MissingRLK {
		t.Error("expected missing_rlk=true")
	}
}

func TestHandleFinalizeMissingGaloisElements(t *testing.T) {
	// Get two valid Galois elements.
	tmpSrv, tmpParams, _ := testSetup(t)
	_ = tmpSrv
	ge1 := tmpParams.GaloisElement(1)
	ge2 := tmpParams.GaloisElement(2)

	srv, ckksParams, sk := testSetupWithGalois(t, []int{int(ge1), int(ge2)})
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, _ := rlk.MarshalBinary()

	// Upload RLK but only ge1 (missing ge2).
	gk1 := kg.GenGaloisKeyNew(ge1, sk)
	gk1Bytes, _ := gk1.MarshalBinary()

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
	uploadGaloisKey(t, handler, id, ge1, gk1Bytes)

	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/keys/finalize", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Result().StatusCode)
	}

	var errResp finalizeErrorResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&errResp); err != nil {
		t.Fatalf("decoding error response: %v", err)
	}
	if len(errResp.MissingElements) != 1 || errResp.MissingElements[0] != ge2 {
		t.Errorf("expected missing_elements=[%d], got %v", ge2, errResp.MissingElements)
	}
}

func TestHandleFinalizeAlreadyReady(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, _ := rlk.MarshalBinary()

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
	finalizeSession(t, handler, id)

	// Second finalize should get 409.
	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/keys/finalize", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusConflict {
		t.Errorf("expected 409 for already-finalized session, got %d", w.Result().StatusCode)
	}
}

func TestHandleInferRejectPendingSession(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := srv.Handler("")

	id := createSession(t, handler)

	// Infer on a pending session should get 409.
	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/infer", bytes.NewReader([]byte("data")))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusConflict {
		t.Errorf("expected 409 for pending session, got %d", w.Result().StatusCode)
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
	handler := srv.Handler("")

	kg := rlwe.NewKeyGenerator(ckksParams)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, _ := rlk.MarshalBinary()

	id := createSession(t, handler)
	uploadRLK(t, handler, id, rlkBytes)
	finalizeSession(t, handler, id)

	// Infer with empty body.
	req := httptest.NewRequest(http.MethodPost, "/session/"+id+"/infer", bytes.NewReader(nil))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Result().StatusCode)
	}
}

func TestFullRoundtrip(t *testing.T) {
	srv, ckksParams, sk := testSetup(t)
	handler := srv.Handler("")

	// Generate keys.
	kg := rlwe.NewKeyGenerator(ckksParams)
	pk := kg.GenPublicKeyNew(sk)
	rlk := kg.GenRelinearizationKeyNew(sk)
	rlkBytes, err := rlk.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary RLK: %v", err)
	}

	// 1. GET /params — verify response format.
	req := httptest.NewRequest(http.MethodGet, "/params", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("GET /params: expected 200, got %d", w.Result().StatusCode)
	}

	// 2. POST /session — create pending session.
	id := createSession(t, handler)

	// 3. Upload RLK.
	uploadRLK(t, handler, id, rlkBytes)

	// 4. Finalize.
	finalizeSession(t, handler, id)

	// 5. Encrypt input — use distinct values to catch slot permutation bugs.
	encoder := ckks.NewEncoder(ckksParams)
	encryptor := ckks.NewEncryptor(ckksParams, pk)
	decryptor := ckks.NewDecryptor(ckksParams, sk)

	inputValues := make([]float64, ckksParams.MaxSlots())
	for i := range inputValues {
		inputValues[i] = float64(i%10) * 0.1 // 0.0, 0.1, 0.2, ..., 0.9, 0.0, ...
	}

	pt := ckks.NewPlaintext(ckksParams, 2) // input_level = 2
	pt.Scale = ckksParams.DefaultScale()
	encoder.Encode(inputValues, pt)
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}

	ctBytes, err := ct.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal CT: %v", err)
	}

	// 6. POST /session/{id}/infer.
	req = httptest.NewRequest(http.MethodPost, "/session/"+id+"/infer", bytes.NewReader(ctBytes))
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		body, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("POST /session/{id}/infer: expected 200, got %d: %s", w.Result().StatusCode, body)
	}

	// 7. Decrypt result.
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

	// Quad = x^2. Check the first 10 slots with their distinct expected values.
	tolerance := 0.01
	for i := 0; i < 10; i++ {
		expected := inputValues[i] * inputValues[i]
		diff := outputValues[i] - expected
		if diff < -tolerance || diff > tolerance {
			t.Errorf("slot %d: expected ~%.4f (%.1f^2), got %.4f (diff=%.6f)",
				i, expected, inputValues[i], outputValues[i], diff)
		}
	}
}

func TestCORSMiddleware(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := corsMiddleware(srv.Handler(""))

	// Regular GET request should have CORS headers.
	req := httptest.NewRequest(http.MethodGet, "/params", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	resp := w.Result()
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("expected Access-Control-Allow-Origin: *, got %q", got)
	}
	if got := resp.Header.Get("Access-Control-Allow-Methods"); got == "" {
		t.Error("missing Access-Control-Allow-Methods header")
	}

	// OPTIONS preflight should return 200 with no body processing.
	req = httptest.NewRequest(http.MethodOptions, "/session", nil)
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	resp = w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("OPTIONS: expected 200, got %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("OPTIONS: expected Access-Control-Allow-Origin: *, got %q", got)
	}
}

func TestKeyUploadToNonexistentSession(t *testing.T) {
	srv, _, _ := testSetup(t)
	handler := srv.Handler("")

	// RLK upload to nonexistent session.
	req := httptest.NewRequest(http.MethodPost, "/session/nonexistent/keys/relin", bytes.NewReader([]byte("data")))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Result().StatusCode)
	}

	// Galois key upload to nonexistent session.
	req = httptest.NewRequest(http.MethodPost, "/session/nonexistent/keys/galois/3", bytes.NewReader([]byte("data")))
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Result().StatusCode)
	}

	// Finalize nonexistent session.
	req = httptest.NewRequest(http.MethodPost, "/session/nonexistent/keys/finalize", nil)
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Result().StatusCode)
	}
}
