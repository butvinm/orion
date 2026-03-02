// Package main implements an HTTP server for the WASM browser demo.
//
// Endpoints:
//   - GET  /params             — CKKS params, key manifest, input level
//   - POST /session            — accepts MemEvaluationKeySet bytes, returns session ID
//   - POST /session/{id}/infer — accepts ciphertext bytes, returns result ciphertext bytes
//   - GET  /                   — static files from client directory
package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"

	"github.com/baahl-nyu/orion/evaluator"
)

// Maximum request body sizes.
const (
	maxKeySetBytes    = 512 * 1024 * 1024 // 512 MB — key sets can be large
	maxCiphertextBytes = 32 * 1024 * 1024  // 32 MB — ciphertexts are much smaller
)

// session holds an evaluator created from client-provided keys.
// mu serializes Forward calls — Lattigo evaluators are not goroutine-safe.
type session struct {
	mu   sync.Mutex
	eval *evaluator.Evaluator
}

// Server holds shared state for the HTTP handlers.
type Server struct {
	model      *evaluator.Model
	ckksParams ckks.Parameters // cached at construction time
	mu         sync.Mutex
	sessions   map[string]*session
}

// NewServer creates a Server with a loaded model.
func NewServer(model *evaluator.Model) (*Server, error) {
	orionParams, _, _ := model.ClientParams()
	ckksParams, err := orionParams.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS params: %w", err)
	}
	return &Server{
		model:      model,
		ckksParams: ckksParams,
		sessions:   make(map[string]*session),
	}, nil
}

// paramsResponse is the JSON response for GET /params.
type paramsResponse struct {
	CKKSParams  json.RawMessage `json:"ckks_params"`
	KeyManifest json.RawMessage `json:"key_manifest"`
	InputLevel  int             `json:"input_level"`
}

// HandleParams returns CKKS parameters, key manifest, and input level.
func (s *Server) HandleParams(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	params, manifest, inputLevel := s.model.ClientParams()

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		http.Error(w, fmt.Sprintf("marshaling params: %v", err), http.StatusInternalServerError)
		return
	}
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		http.Error(w, fmt.Sprintf("marshaling manifest: %v", err), http.StatusInternalServerError)
		return
	}

	resp := paramsResponse{
		CKKSParams:  paramsJSON,
		KeyManifest: manifestJSON,
		InputLevel:  inputLevel,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("HandleParams: encode response: %v", err)
	}
}

// sessionResponse is the JSON response for POST /session.
type sessionResponse struct {
	SessionID string `json:"session_id"`
}

// HandleSession creates a new session from uploaded evaluation keys.
func (s *Server) HandleSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxKeySetBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body: expected MemEvaluationKeySet bytes", http.StatusBadRequest)
		return
	}

	// Unmarshal evaluation keys.
	evk := &rlwe.MemEvaluationKeySet{}
	if err := evk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling evaluation keys: %v", err), http.StatusBadRequest)
		return
	}

	// Create evaluator using cached CKKS params.
	eval, err := evaluator.NewEvaluatorFromKeySet(s.ckksParams, evk)
	if err != nil {
		http.Error(w, fmt.Sprintf("creating evaluator: %v", err), http.StatusInternalServerError)
		return
	}

	// Generate a random session ID.
	id, err := newSessionID()
	if err != nil {
		http.Error(w, "generating session ID", http.StatusInternalServerError)
		return
	}

	// Store session.
	s.mu.Lock()
	s.sessions[id] = &session{eval: eval}
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(sessionResponse{SessionID: id}); err != nil {
		log.Printf("HandleSession: encode response: %v", err)
	}
}

// HandleInfer runs inference on a ciphertext using the session's evaluator.
func (s *Server) HandleInfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract session ID from path: /session/{id}/infer
	// Using Go 1.22+ ServeMux pattern matching.
	sessionID := r.PathValue("id")
	if sessionID == "" {
		http.Error(w, "missing session ID", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	sess, ok := s.sessions[sessionID]
	s.mu.Unlock()
	if !ok {
		http.Error(w, fmt.Sprintf("session %q not found", sessionID), http.StatusNotFound)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxCiphertextBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body: expected ciphertext bytes", http.StatusBadRequest)
		return
	}

	// Unmarshal input ciphertext.
	ct := &rlwe.Ciphertext{}
	if err := ct.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling ciphertext: %v", err), http.StatusBadRequest)
		return
	}

	// Serialize Forward calls on this session — Lattigo evaluators are not goroutine-safe.
	sess.mu.Lock()
	result, err := sess.eval.Forward(s.model, ct)
	sess.mu.Unlock()
	if err != nil {
		http.Error(w, fmt.Sprintf("inference error: %v", err), http.StatusInternalServerError)
		return
	}

	// Marshal result.
	resultBytes, err := result.MarshalBinary()
	if err != nil {
		http.Error(w, fmt.Sprintf("marshaling result: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	if _, err := w.Write(resultBytes); err != nil {
		log.Printf("HandleInfer: write result: %v", err)
	}
}

// Handler returns an http.Handler with all routes registered.
func (s *Server) Handler(clientDir string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /params", s.HandleParams)
	mux.HandleFunc("POST /session", s.HandleSession)
	mux.HandleFunc("POST /session/{id}/infer", s.HandleInfer)

	if clientDir != "" {
		mux.Handle("GET /", http.FileServer(http.Dir(clientDir)))
	}

	return mux
}

// newSessionID generates a cryptographically random session ID.
func newSessionID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func main() {
	modelPath := "model.orion"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	clientDir := "../client"
	if len(os.Args) > 2 {
		clientDir = os.Args[2]
	}

	addr := ":8080"
	if len(os.Args) > 3 {
		addr = os.Args[3]
	}

	log.Printf("Loading model from %s...", modelPath)
	data, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("reading model: %v", err)
	}

	model, err := evaluator.LoadModel(data)
	if err != nil {
		log.Fatalf("loading model: %v", err)
	}

	params, manifest, inputLevel := model.ClientParams()
	log.Printf("Model loaded: logn=%d, galois_elements=%d, needs_rlk=%v, input_level=%d",
		params.LogN, len(manifest.GaloisElements), manifest.NeedsRLK, inputLevel)

	srv, err := NewServer(model)
	if err != nil {
		log.Fatalf("creating server: %v", err)
	}

	// Enable CORS for browser demo.
	handler := corsMiddleware(srv.Handler(clientDir))

	log.Printf("Server listening on %s", addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// corsMiddleware adds CORS headers for browser access.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// CKKSParams returns the cached CKKS parameters (exposed for testing).
func (s *Server) CKKSParams() ckks.Parameters {
	return s.ckksParams
}
