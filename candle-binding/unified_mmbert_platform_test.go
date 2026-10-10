package candle_binding

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestUnifiedMMBertRefusedBeforePreparation(t *testing.T) {
	for _, name := range []string{"mmbert", "mmbert32k", "mmbert-32k"} {
		err := unifiedMMBertRefused("darwin", "arm64", name)
		if !errors.Is(err, ErrUnifiedMMBertUnsupported) {
			t.Fatalf("%s: got %v", name, err)
		}
	}
	if err := unifiedMMBertRefused("darwin", "arm64", "modernbert"); err != nil {
		t.Fatalf("ordinary ModernBERT architecture was refused: %v", err)
	}

	dir := t.TempDir()
	mmbertConfig := `{"model_type":"modernbert","vocab_size":256000,"position_embedding_type":"sans_pos","max_position_embeddings":32768}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(mmbertConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("test must not create model weights")
	}
	if err := unifiedMMBertRefused("darwin", "arm64", "modernbert", dir, dir, dir); !errors.Is(err, ErrUnifiedMMBertUnsupported) {
		t.Fatalf("mmBERT checkpoint was not refused: %v", err)
	}
	if err := unifiedMMBertRefused("linux", "arm64", "mmbert32k", dir); err != nil {
		t.Fatalf("linux arm64 must keep preparing unified mmBERT: %v", err)
	}
	ordinary := `{"model_type":"modernbert","vocab_size":50368,"position_embedding_type":"sans_pos","max_position_embeddings":8192}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(ordinary), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := unifiedMMBertRefused("darwin", "arm64", "modernbert", dir); err != nil {
		t.Fatalf("ordinary ModernBERT unified init must stay available: %v", err)
	}
}
