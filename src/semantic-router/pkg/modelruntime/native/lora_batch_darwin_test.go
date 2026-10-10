package native

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const loraBatchChildEnv = "VLLM_SR_LORA_BATCH_CHILD"

// TestLoRABatchUnifiedMMBertInitializationIsBounded covers the production
// initializer used by UnifiedClassifier.initializeLoRABindings. It must
// return the explicit platform refusal before any native preparation, rather
// than relying on a timeout which cannot unblock the native wait.
//
// The load runs in a child process. The parent enforces the deadline and kills
// the child if native preparation never returns. A check after a synchronous
// call cannot do that: a hang never reaches the check, and a Go test timeout
// cannot interrupt threads already parked inside the native call.
func TestLoRABatchUnifiedMMBertInitializationIsBounded(t *testing.T) {
	if os.Getenv(loraBatchChildEnv) == "1" {
		runLoRABatchBoundChild(t)
		return
	}
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("production LoRABatch bound is measured on darwin/arm64")
	}
	intent, pii, guard := downloadedMMBertDirs(t)
	for _, dir := range []string{intent, pii, guard} {
		if _, statErr := os.Stat(filepath.Join(dir, "model.safetensors")); statErr != nil {
			t.Skipf("merged mmBERT weights missing at %s: %v", dir, statErr)
		}
	}
	err := runChildUntil(t, 5*time.Second, map[string]string{
		loraBatchChildEnv:     "1",
		"VLLM_SR_LORA_INTENT": intent,
		"VLLM_SR_LORA_PII":    pii,
		"VLLM_SR_LORA_GUARD":  guard,
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestParentDeadlineKillsHungChild proves the parent can fail a child that
// never returns, without waiting for the Go test timeout.
func TestParentDeadlineKillsHungChild(t *testing.T) {
	if os.Getenv(loraBatchChildEnv) == "hang" {
		time.Sleep(time.Hour)
		return
	}
	err := runChildUntil(t, 500*time.Millisecond, map[string]string{loraBatchChildEnv: "hang"})
	if err == nil || !strings.Contains(err.Error(), "parent killed it") {
		t.Fatalf("expected parent to kill the hung child, got %v", err)
	}
}

func runChildUntil(t *testing.T, deadline time.Duration, extra map[string]string) error {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$", "-test.count=1")
	cmd.Env = append(os.Environ(), envList(extra)...)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("child %s was not bounded: %w\n%s", t.Name(), err, output.String())
		}
		return nil
	case <-time.After(deadline):
		_ = cmd.Process.Kill()
		<-done
		return fmt.Errorf("child %s exceeded %s; parent killed it\n%s", t.Name(), deadline, output.String())
	}
}

func runLoRABatchBoundChild(t *testing.T) {
	t.Helper()
	specs := loraBatchModelSpecs(os.Getenv("VLLM_SR_LORA_INTENT"), os.Getenv("VLLM_SR_LORA_PII"), os.Getenv("VLLM_SR_LORA_GUARD"))
	batch, err := New(nil).LoRABatch(context.Background(), specs[0], specs[1], specs[2])
	if !errors.Is(err, candle.ErrUnifiedMMBertUnsupported) {
		if batch != nil {
			_ = batch.Close()
		}
		t.Fatalf("owned mmBERT initialization was not refused before native loading: %v", err)
	}
}

func downloadedMMBertDirs(t *testing.T) (intent, pii, guard string) {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "..", "models"))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(root, "mmbert32k-intent-classifier-merged"),
		filepath.Join(root, "mmbert32k-pii-detector-merged"),
		filepath.Join(root, "mmbert32k-jailbreak-detector-merged")
}

func envList(extra map[string]string) []string {
	values := make([]string, 0, len(extra))
	for key, value := range extra {
		values = append(values, key+"="+value)
	}
	return values
}

func loraBatchModelSpecs(intent, pii, guard string) [3]config.ResolvedModelBinding {
	paths := [3]string{intent, pii, guard}
	contracts := [3]string{
		config.RemoteClassifierContractLabelDistribution,
		config.RemoteClassifierContractTokenSpans,
		config.RemoteClassifierContractLabelDistribution,
	}
	var specs [3]config.ResolvedModelBinding
	for i, contract := range contracts {
		specs[i] = config.ResolvedModelBinding{
			Recipe: "test",
			Name:   contract,
			Binding: config.ModelBinding{
				Deployment: "local",
				Adapter:    "modernbert",
				Contract:   contract,
			},
			Deployment: config.ModelDeployment{
				Provider:  "candle",
				Artifact:  paths[i],
				Device:    "cpu",
				Precision: "native",
				Input:     config.ModelInputBudget{MaxTokens: 128, Overflow: "truncate"},
			},
		}
	}
	return specs
}
