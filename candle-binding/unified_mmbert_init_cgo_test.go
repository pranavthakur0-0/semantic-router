//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestInitLoRAUnifiedClassifierRefusesOnThisDarwinArm64Host(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("InitLoRAUnifiedClassifier follows the host platform; injected cases cover the refusal")
	}
	err := InitLoRAUnifiedClassifier("unused-intent", "unused-pii", "unused-security", "mmbert32k", true)
	if !errors.Is(err, ErrUnifiedMMBertUnsupported) {
		t.Fatalf("native preparation was not refused: %v", err)
	}
}

func TestOwnedInstanceLoadRefusesBeforeNativePreparation(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("owned instance loading follows the host platform; injected cases cover the refusal")
	}
	dir := t.TempDir()
	config := `{"model_type":"modernbert","vocab_size":256000,"position_embedding_type":"sans_pos"}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSequenceClassifier(InstanceOptions{ModelPath: dir, ModelType: "modernbert"}); !errors.Is(err, ErrUnifiedMMBertUnsupported) {
		t.Fatalf("owned instance preparation entered native loading: %v", err)
	}
}

func TestInitLoRAUnifiedClassifierRefusesDownloadedWeightsBeforeLoad(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("requires darwin/arm64")
	}
	root, err := filepath.Abs(filepath.Join("..", "models"))
	if err != nil {
		t.Fatal(err)
	}
	intent := filepath.Join(root, "mmbert32k-intent-classifier-merged")
	pii := filepath.Join(root, "mmbert32k-pii-detector-merged")
	guard := filepath.Join(root, "mmbert32k-jailbreak-detector-merged")
	for _, dir := range []string{intent, pii, guard} {
		if _, statErr := os.Stat(filepath.Join(dir, "model.safetensors")); statErr != nil {
			t.Skipf("merged mmBERT weights missing at %s: %v", dir, statErr)
		}
	}
	start := time.Now()
	err = InitLoRAUnifiedClassifier(intent, pii, guard, "mmbert32k", true)
	if !errors.Is(err, ErrUnifiedMMBertUnsupported) {
		t.Fatalf("real weights entered native init: %v (after %v)", err, time.Since(start))
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("refusal took %v; expected immediate return before native load", time.Since(start))
	}
}
