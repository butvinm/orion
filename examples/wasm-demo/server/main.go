// Package main implements an HTTP server for the WASM browser demo.
//
// Endpoints:
//   - GET  /params                            — CKKS params, key manifest, input level
//   - POST /session                           — creates pending session, returns session ID
//   - POST /session/{id}/keys/relin           — uploads relinearization key
//   - POST /session/{id}/keys/galois/{element} — uploads individual Galois key
//   - POST /session/{id}/keys/bootstrap       — uploads bootstrap evaluation keys
//   - POST /session/{id}/keys/finalize        — validates keys, creates evaluator
//   - POST /session/{id}/infer                — accepts ciphertext bytes, returns result
//   - GET  /                                  — static files from client directory
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	orion "github.com/butvinm/orion/v2"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/butvinm/orion/v2/evaluator"
)

// Maximum request body sizes.
const (
	maxSingleKeyBytes      = 16 * 1024 * 1024           // 16 MB — individual key upload
	maxCiphertextBytes     = 32 * 1024 * 1024            // 32 MB — ciphertexts are much smaller
	maxBootstrapKeyBytes   = 6 * 1024 * 1024 * 1024      // 6 GB — bootstrap keys are much larger than individual Galois keys
)

// sessionState represents the state of a session in the key upload state machine.
type sessionState int

const (
	sessionPending sessionState = iota // keys being uploaded
	sessionReady                       // evaluator created, ready for inference
)

// session holds per-session state during key upload and inference.
type session struct {
	mu           sync.Mutex
	state        sessionState
	lastActivity time.Time

	// Key storage (populated during pending state).
	rlk        *rlwe.RelinearizationKey
	galoisKeys map[uint64]*rlwe.GaloisKey
	btpKeys    *bootstrapping.EvaluationKeys

	// Evaluator (populated after finalize).
	eval *evaluator.Evaluator
}

// Default session timeout for pending sessions with no activity.
const defaultSessionTimeout = 5 * time.Minute

// Server holds shared state for the HTTP handlers.
type Server struct {
	model          *evaluator.Model
	ckksParams     ckks.Parameters // cached at construction time
	manifest       orion.Manifest  // cached at construction time
	sessionTimeout time.Duration   // inactivity timeout for pending sessions
	mu             sync.Mutex
	sessions       map[string]*session
}

// NewServer creates a Server with a loaded model.
func NewServer(model *evaluator.Model) (*Server, error) {
	orionParams, manifest, _ := model.ClientParams()
	ckksParams, err := orionParams.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS params: %w", err)
	}
	return &Server{
		model:          model,
		ckksParams:     ckksParams,
		manifest:       manifest,
		sessionTimeout: defaultSessionTimeout,
		sessions:       make(map[string]*session),
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

// HandleSession creates a new pending session (no body required).
func (s *Server) HandleSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	id, err := newSessionID()
	if err != nil {
		http.Error(w, "generating session ID", http.StatusInternalServerError)
		return
	}

	s.mu.Lock()
	s.sessions[id] = &session{
		state:        sessionPending,
		lastActivity: time.Now(),
		galoisKeys:   make(map[uint64]*rlwe.GaloisKey),
	}
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(sessionResponse{SessionID: id}); err != nil {
		log.Printf("HandleSession: encode response: %v", err)
	}
}

// getSession looks up a session by ID from the request path.
func (s *Server) getSession(w http.ResponseWriter, r *http.Request) (*session, bool) {
	sessionID := r.PathValue("id")
	if sessionID == "" {
		http.Error(w, "missing session ID", http.StatusBadRequest)
		return nil, false
	}

	s.mu.Lock()
	sess, ok := s.sessions[sessionID]
	s.mu.Unlock()
	if !ok {
		http.Error(w, fmt.Sprintf("session %q not found", sessionID), http.StatusNotFound)
		return nil, false
	}
	return sess, true
}

// HandleRelinKey uploads a relinearization key for a pending session.
func (s *Server) HandleRelinKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}

	sess.mu.Lock()
	defer sess.mu.Unlock()

	if sess.state == sessionReady {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxSingleKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body: expected RelinearizationKey bytes", http.StatusBadRequest)
		return
	}

	rlk := &rlwe.RelinearizationKey{}
	if err := rlk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling relinearization key: %v", err), http.StatusBadRequest)
		return
	}

	sess.rlk = rlk
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

// HandleGaloisKey uploads an individual Galois key for a pending session.
func (s *Server) HandleGaloisKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}

	elementStr := r.PathValue("element")
	element, err := strconv.ParseUint(elementStr, 10, 64)
	if err != nil {
		http.Error(w, fmt.Sprintf("invalid Galois element %q: %v", elementStr, err), http.StatusBadRequest)
		return
	}

	sess.mu.Lock()
	defer sess.mu.Unlock()

	if sess.state == sessionReady {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxSingleKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body: expected GaloisKey bytes", http.StatusBadRequest)
		return
	}

	gk := &rlwe.GaloisKey{}
	if err := gk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling Galois key: %v", err), http.StatusBadRequest)
		return
	}

	sess.galoisKeys[element] = gk
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

// HandleBootstrapKey uploads bootstrap evaluation keys for a pending session.
func (s *Server) HandleBootstrapKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}

	// Quick state check before expensive body read to reject finalized sessions early.
	sess.mu.Lock()
	ready := sess.state == sessionReady
	sess.mu.Unlock()
	if ready {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}

	// Read and unmarshal body without holding session lock to avoid blocking
	// other operations during a potentially slow multi-GB network read.
	r.Body = http.MaxBytesReader(w, r.Body, maxBootstrapKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body: expected bootstrap EvaluationKeys bytes", http.StatusBadRequest)
		return
	}

	btpKeys := &bootstrapping.EvaluationKeys{}
	if err := btpKeys.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling bootstrap keys: %v", err), http.StatusBadRequest)
		return
	}

	sess.mu.Lock()
	defer sess.mu.Unlock()

	if sess.state == sessionReady {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}

	sess.btpKeys = btpKeys
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

// finalizeErrorResponse is the JSON response for finalize validation errors.
type finalizeErrorResponse struct {
	Error              string   `json:"error"`
	MissingRLK         bool     `json:"missing_rlk,omitempty"`
	MissingElements    []uint64 `json:"missing_elements,omitempty"`
	MissingBtpKeys     bool     `json:"missing_btp_keys,omitempty"`
}

// HandleFinalize validates all required keys are present and creates the evaluator.
func (s *Server) HandleFinalize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}

	sess.mu.Lock()
	defer sess.mu.Unlock()

	if sess.state == sessionReady {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}

	// Validate completeness against cached manifest.
	var missingElements []uint64
	for _, ge := range s.manifest.GaloisElements {
		if _, ok := sess.galoisKeys[ge]; !ok {
			missingElements = append(missingElements, ge)
		}
	}

	missingRLK := s.manifest.NeedsRLK && sess.rlk == nil
	missingBtpKeys := len(s.manifest.BootstrapSlots) > 0 && sess.btpKeys == nil

	if missingRLK || len(missingElements) > 0 || missingBtpKeys {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		if err := json.NewEncoder(w).Encode(finalizeErrorResponse{
			Error:           "incomplete keys",
			MissingRLK:      missingRLK,
			MissingElements: missingElements,
			MissingBtpKeys:  missingBtpKeys,
		}); err != nil {
			log.Printf("HandleFinalize: encode error response: %v", err)
		}
		return
	}

	// Assemble MemEvaluationKeySet.
	galKeys := make([]*rlwe.GaloisKey, 0, len(sess.galoisKeys))
	for _, gk := range sess.galoisKeys {
		galKeys = append(galKeys, gk)
	}
	evk := rlwe.NewMemEvaluationKeySet(sess.rlk, galKeys...)

	eval, err := evaluator.NewEvaluatorFromKeySet(s.ckksParams, evk, sess.btpKeys)
	if err != nil {
		http.Error(w, fmt.Sprintf("creating evaluator: %v", err), http.StatusInternalServerError)
		return
	}

	sess.eval = eval
	sess.state = sessionReady
	// Free key storage — no longer needed.
	sess.rlk = nil
	sess.galoisKeys = nil
	sess.btpKeys = nil
	sess.lastActivity = time.Now()

	w.WriteHeader(http.StatusOK)
}

// HandleInfer runs inference on a ciphertext using the session's evaluator.
func (s *Server) HandleInfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}

	sess.mu.Lock()
	if sess.state != sessionReady {
		sess.mu.Unlock()
		http.Error(w, "session not finalized: upload keys and call /keys/finalize first", http.StatusConflict)
		return
	}
	sess.mu.Unlock()

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

// StartCleanup launches a background goroutine that periodically removes
// pending sessions that have been inactive for longer than sessionTimeout.
// Ready sessions are never cleaned up. The goroutine stops when ctx is cancelled.
func (s *Server) StartCleanup(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.cleanupExpired()
			}
		}
	}()
}

// cleanupExpired removes sessions whose lastActivity exceeds sessionTimeout.
// Both pending and ready sessions are cleaned up to prevent memory leaks.
func (s *Server) cleanupExpired() {
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, sess := range s.sessions {
		sess.mu.Lock()
		expired := now.Sub(sess.lastActivity) > s.sessionTimeout
		eval := sess.eval
		sess.mu.Unlock()
		if expired {
			log.Printf("cleaning up expired session %s", id)
			if eval != nil {
				eval.Close()
			}
			delete(s.sessions, id)
		}
	}
}

// Handler returns an http.Handler with all routes registered.
func (s *Server) Handler(clientDir string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /params", s.HandleParams)
	mux.HandleFunc("POST /session", s.HandleSession)
	mux.HandleFunc("POST /session/{id}/keys/relin", s.HandleRelinKey)
	mux.HandleFunc("POST /session/{id}/keys/galois/{element}", s.HandleGaloisKey)
	mux.HandleFunc("POST /session/{id}/keys/bootstrap", s.HandleBootstrapKey)
	mux.HandleFunc("POST /session/{id}/keys/finalize", s.HandleFinalize)
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
	modelPath := "../model.orion"
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

	// Start background cleanup of stale pending sessions.
	srv.StartCleanup(context.Background(), 30*time.Second)

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
