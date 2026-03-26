// Package main implements an HTTP server for the C3AE age verification FHE demo.
//
// Reuses the same session-based architecture as the wasm-demo server.
// Endpoints mirror the wasm-demo with the same key upload / finalize / infer flow.
package main

import (
	"context"
	"crypto/rand"
	"encoding/binary"
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

const (
	maxSingleKeyBytes    = 16 * 1024 * 1024       // 16 MB
	maxBootstrapSingleKeyBytes = 256 * 1024 * 1024       // 256 MB — individual bootstrap keys are larger than main keys
	maxCiphertextBytes         = 32 * 1024 * 1024        // 32 MB
	maxBootstrapKeyBytes       = 6 * 1024 * 1024 * 1024  // 6 GB — monolithic bootstrap blob
)

type sessionState int

const (
	sessionPending sessionState = iota
	sessionReady
)

type session struct {
	mu           sync.Mutex
	state        sessionState
	lastActivity time.Time

	// Main evaluation keys
	rlk        *rlwe.RelinearizationKey
	galoisKeys map[uint64]*rlwe.GaloisKey

	// Bootstrap keys (streamed individually)
	btpKeys       *bootstrapping.EvaluationKeys // set if uploaded as monolithic blob
	btpRLK        *rlwe.RelinearizationKey
	btpGaloisKeys map[uint64]*rlwe.GaloisKey
	btpSwitchKeys map[string]*rlwe.EvaluationKey // "EvkN1ToN2", "EvkDenseToSparse", etc.

	eval *evaluator.Evaluator
}

const defaultSessionTimeout = 10 * time.Minute

type Server struct {
	model          *evaluator.Model
	ckksParams     ckks.Parameters
	manifest       orion.Manifest
	sessionTimeout time.Duration
	mu             sync.Mutex
	sessions       map[string]*session
}

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

type paramsResponse struct {
	CKKSParams  json.RawMessage `json:"ckks_params"`
	KeyManifest json.RawMessage `json:"key_manifest"`
	InputLevel  int             `json:"input_level"`
}

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

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(paramsResponse{
		CKKSParams:  paramsJSON,
		KeyManifest: manifestJSON,
		InputLevel:  inputLevel,
	})
}

type sessionResponse struct {
	SessionID string `json:"session_id"`
}

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
		state:         sessionPending,
		lastActivity:  time.Now(),
		galoisKeys:    make(map[uint64]*rlwe.GaloisKey),
		btpGaloisKeys: make(map[uint64]*rlwe.GaloisKey),
		btpSwitchKeys: make(map[string]*rlwe.EvaluationKey),
	}
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sessionResponse{SessionID: id})
}

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
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}

	rlk := &rlwe.RelinearizationKey{}
	if err := rlk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling RLK: %v", err), http.StatusBadRequest)
		return
	}

	sess.rlk = rlk
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

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
		http.Error(w, fmt.Sprintf("invalid Galois element %q", elementStr), http.StatusBadRequest)
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
		http.Error(w, "empty body", http.StatusBadRequest)
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

func (s *Server) HandleBootstrapKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}

	sess.mu.Lock()
	ready := sess.state == sessionReady
	sess.mu.Unlock()
	if ready {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxBootstrapKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body", http.StatusBadRequest)
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

// HandleBootstrapRelinKey uploads the bootstrap relinearization key.
func (s *Server) HandleBootstrapRelinKey(w http.ResponseWriter, r *http.Request) {
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
	r.Body = http.MaxBytesReader(w, r.Body, maxBootstrapSingleKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}
	rlk := &rlwe.RelinearizationKey{}
	if err := rlk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling bootstrap RLK: %v", err), http.StatusBadRequest)
		return
	}
	sess.btpRLK = rlk
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

// HandleBootstrapGaloisKey uploads an individual bootstrap Galois key.
func (s *Server) HandleBootstrapGaloisKey(w http.ResponseWriter, r *http.Request) {
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
		http.Error(w, fmt.Sprintf("invalid element %q", elementStr), http.StatusBadRequest)
		return
	}
	sess.mu.Lock()
	defer sess.mu.Unlock()
	if sess.state == sessionReady {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBootstrapSingleKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}
	gk := &rlwe.GaloisKey{}
	if err := gk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling bootstrap Galois key: %v", err), http.StatusBadRequest)
		return
	}
	sess.btpGaloisKeys[element] = gk
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

// HandleBootstrapSwitchingKey uploads a named bootstrap switching/evaluation key.
func (s *Server) HandleBootstrapSwitchingKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	sess, ok := s.getSession(w, r)
	if !ok {
		return
	}
	name := r.PathValue("name")
	if name == "" {
		http.Error(w, "missing key name", http.StatusBadRequest)
		return
	}
	sess.mu.Lock()
	defer sess.mu.Unlock()
	if sess.state == sessionReady {
		http.Error(w, "session already finalized", http.StatusConflict)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBootstrapSingleKeyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("reading body: %v", err), http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}
	evk := &rlwe.EvaluationKey{}
	if err := evk.UnmarshalBinary(body); err != nil {
		http.Error(w, fmt.Sprintf("unmarshaling switching key %q: %v", name, err), http.StatusBadRequest)
		return
	}
	sess.btpSwitchKeys[name] = evk
	sess.lastActivity = time.Now()
	w.WriteHeader(http.StatusOK)
}

type finalizeErrorResponse struct {
	Error           string   `json:"error"`
	MissingRLK      bool     `json:"missing_rlk,omitempty"`
	MissingElements []uint64 `json:"missing_elements,omitempty"`
	MissingBtpKeys  bool     `json:"missing_btp_keys,omitempty"`
}

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

	var missingElements []uint64
	for _, ge := range s.manifest.GaloisElements {
		if _, ok := sess.galoisKeys[ge]; !ok {
			missingElements = append(missingElements, ge)
		}
	}

	missingRLK := s.manifest.NeedsRLK && sess.rlk == nil
	// Bootstrap keys can come as monolithic blob OR streamed individually
	hasBtpBlob := sess.btpKeys != nil
	hasBtpStreamed := sess.btpRLK != nil || len(sess.btpGaloisKeys) > 0
	missingBtpKeys := len(s.manifest.BootstrapSlots) > 0 && !hasBtpBlob && !hasBtpStreamed

	if missingRLK || len(missingElements) > 0 || missingBtpKeys {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(finalizeErrorResponse{
			Error:           "incomplete keys",
			MissingRLK:      missingRLK,
			MissingElements: missingElements,
			MissingBtpKeys:  missingBtpKeys,
		})
		return
	}

	// Assemble main evaluation key set
	galKeys := make([]*rlwe.GaloisKey, 0, len(sess.galoisKeys))
	for _, gk := range sess.galoisKeys {
		galKeys = append(galKeys, gk)
	}
	evk := rlwe.NewMemEvaluationKeySet(sess.rlk, galKeys...)

	// Assemble bootstrap keys
	var btpKeys *bootstrapping.EvaluationKeys
	if hasBtpBlob {
		btpKeys = sess.btpKeys
	} else if hasBtpStreamed {
		// Build from individually uploaded keys
		btpGalKeys := make([]*rlwe.GaloisKey, 0, len(sess.btpGaloisKeys))
		for _, gk := range sess.btpGaloisKeys {
			btpGalKeys = append(btpGalKeys, gk)
		}
		btpKeys = &bootstrapping.EvaluationKeys{
			MemEvaluationKeySet: rlwe.NewMemEvaluationKeySet(sess.btpRLK, btpGalKeys...),
		}
		// Attach switching keys by name
		for name, key := range sess.btpSwitchKeys {
			switch name {
			case "EvkN1ToN2":
				btpKeys.EvkN1ToN2 = key
			case "EvkN2ToN1":
				btpKeys.EvkN2ToN1 = key
			case "EvkRealToCmplx":
				btpKeys.EvkRealToCmplx = key
			case "EvkCmplxToReal":
				btpKeys.EvkCmplxToReal = key
			case "EvkDenseToSparse":
				btpKeys.EvkDenseToSparse = key
			case "EvkSparseToDense":
				btpKeys.EvkSparseToDense = key
			}
		}
	}

	eval, err := evaluator.NewEvaluatorFromKeySet(s.ckksParams, evk, btpKeys)
	if err != nil {
		http.Error(w, fmt.Sprintf("creating evaluator: %v", err), http.StatusInternalServerError)
		return
	}

	sess.eval = eval
	sess.state = sessionReady
	sess.rlk = nil
	sess.galoisKeys = nil
	sess.btpKeys = nil
	sess.btpRLK = nil
	sess.btpGaloisKeys = nil
	sess.btpSwitchKeys = nil
	sess.lastActivity = time.Now()

	w.WriteHeader(http.StatusOK)
}

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
		http.Error(w, "session not finalized", http.StatusConflict)
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
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}

	// Parse length-prefixed ciphertext list: [u32 count][u64 len][bytes]...
	if len(body) < 4 {
		http.Error(w, "body too short for CT count", http.StatusBadRequest)
		return
	}
	numCTs := int(binary.LittleEndian.Uint32(body[:4]))
	off := 4
	inputs := make([]*rlwe.Ciphertext, numCTs)
	for i := 0; i < numCTs; i++ {
		if off+8 > len(body) {
			http.Error(w, fmt.Sprintf("truncated CT header at index %d", i), http.StatusBadRequest)
			return
		}
		ctLen := int(binary.LittleEndian.Uint64(body[off : off+8]))
		off += 8
		if off+ctLen > len(body) {
			http.Error(w, fmt.Sprintf("truncated CT data at index %d", i), http.StatusBadRequest)
			return
		}
		ct := &rlwe.Ciphertext{}
		if err := ct.UnmarshalBinary(body[off : off+ctLen]); err != nil {
			http.Error(w, fmt.Sprintf("unmarshaling CT %d: %v", i, err), http.StatusBadRequest)
			return
		}
		inputs[i] = ct
		off += ctLen
	}

	t0 := time.Now()
	sess.mu.Lock()
	results, err := sess.eval.Forward(s.model, inputs)
	sess.mu.Unlock()
	elapsed := time.Since(t0)
	if err != nil {
		http.Error(w, fmt.Sprintf("inference error: %v", err), http.StatusInternalServerError)
		return
	}
	log.Printf("FHE inference: %v (%d input CTs -> %d output CTs)", elapsed, numCTs, len(results))

	// Serialize results as length-prefixed list
	var buf []byte
	tmp := make([]byte, 4)
	binary.LittleEndian.PutUint32(tmp, uint32(len(results)))
	buf = append(buf, tmp...)
	for _, r := range results {
		rb, err := r.MarshalBinary()
		if err != nil {
			http.Error(w, fmt.Sprintf("marshaling result: %v", err), http.StatusInternalServerError)
			return
		}
		tmp := make([]byte, 8)
		binary.LittleEndian.PutUint64(tmp, uint64(len(rb)))
		buf = append(buf, tmp...)
		buf = append(buf, rb...)
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Inference-Time", elapsed.String())
	w.Write(buf)
}

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

func (s *Server) Handler(clientDir string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /params", s.HandleParams)
	mux.HandleFunc("POST /session", s.HandleSession)
	mux.HandleFunc("POST /session/{id}/keys/relin", s.HandleRelinKey)
	mux.HandleFunc("POST /session/{id}/keys/galois/{element}", s.HandleGaloisKey)
	mux.HandleFunc("POST /session/{id}/keys/bootstrap", s.HandleBootstrapKey)
	mux.HandleFunc("POST /session/{id}/keys/bootstrap/relin", s.HandleBootstrapRelinKey)
	mux.HandleFunc("POST /session/{id}/keys/bootstrap/galois/{element}", s.HandleBootstrapGaloisKey)
	mux.HandleFunc("POST /session/{id}/keys/bootstrap/switching/{name}", s.HandleBootstrapSwitchingKey)
	mux.HandleFunc("POST /session/{id}/keys/finalize", s.HandleFinalize)
	mux.HandleFunc("POST /session/{id}/infer", s.HandleInfer)

	if clientDir != "" {
		mux.Handle("GET /", http.FileServer(http.Dir(clientDir)))
	}

	return mux
}



func newSessionID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

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
	log.Printf("Model loaded: logn=%d, galois_elements=%d, needs_rlk=%v, bootstrap_slots=%v, input_level=%d",
		params.LogN, len(manifest.GaloisElements), manifest.NeedsRLK, manifest.BootstrapSlots, inputLevel)

	srv, err := NewServer(model)
	if err != nil {
		log.Fatalf("creating server: %v", err)
	}

	srv.StartCleanup(context.Background(), 30*time.Second)

	handler := corsMiddleware(srv.Handler(clientDir))

	log.Printf("Server listening on %s", addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
