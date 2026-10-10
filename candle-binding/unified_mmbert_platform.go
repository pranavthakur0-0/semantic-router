package candle_binding

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// unifiedMMBertNamed is an explicit mmBERT adapter name. "modernbert" is not
// included: ordinary ModernBERT and mmBERT share that config model_type, and
// ModernBertVariant separates them by vocabulary size.
func unifiedMMBertNamed(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "mmbert", "mmbert32k", "mmbert-32k":
		return true
	default:
		return false
	}
}

// unifiedMMBertRefused reports issue #2440 before the legacy native
// initializer runs. goos/goarch are injected so the regression runs on every
// host. Only config.json is read; model weights are left untouched.
//
// A checkpoint is mmBERT when its name says so, or when config.json matches
// ModernBertVariant: vocab_size >= 200000 and position_embedding_type sans_pos.
// A plain modernbert model_type is not enough.
func unifiedMMBertRefused(goos, goarch, architecture string, modelPaths ...string) error {
	if goos != "darwin" || goarch != "arm64" {
		return nil
	}
	if unifiedMMBertNamed(architecture) {
		return ErrUnifiedMMBertUnsupported
	}
	for _, modelPath := range modelPaths {
		if unifiedConfigIsMMBert(modelPath) {
			return ErrUnifiedMMBertUnsupported
		}
	}
	return nil
}

// ValidateUnifiedMMBertClassifierPlatform rejects the classifier configuration
// that can deadlock during native preparation on darwin/arm64. It reads at
// most config.json and never model weights.
func ValidateUnifiedMMBertClassifierPlatform(architecture string, modelPaths ...string) error {
	return unifiedMMBertRefused(runtime.GOOS, runtime.GOARCH, architecture, modelPaths...)
}

func unifiedConfigIsMMBert(modelPath string) bool {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return false
	}
	var meta struct {
		ModelType             string `json:"model_type"`
		VocabSize             int    `json:"vocab_size"`
		PositionEmbeddingType string `json:"position_embedding_type"`
	}
	if err = json.Unmarshal(data, &meta); err != nil {
		return false
	}
	if unifiedMMBertNamed(meta.ModelType) {
		return true
	}
	return meta.VocabSize >= 200000 && meta.PositionEmbeddingType == "sans_pos"
}
