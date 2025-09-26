package all_minilm_l6_v2_test

import (
	"math"
	"testing"

	"github.com/clems4ever/all-minilm-l6-v2-go/all_minilm_l6_v2"
)

func TestSingleSentenceEmbedding(t *testing.T) {
	model, err := all_minilm_l6_v2.NewModel()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	sentence := "Hello, world! This is a test sentence."
	embedding, err := model.Compute(sentence)
	if err != nil {
		t.Fatalf("Failed to compute embedding: %v", err)
	}

	// Check dimensions
	expectedDim := 384
	if len(embedding) != expectedDim {
		t.Errorf("Expected embedding dimension %d, got %d", expectedDim, len(embedding))
	}

	// Verify embedding is not all zeros
	allZeros := true
	for _, val := range embedding {
		if val != 0.0 {
			allZeros = false
			break
		}
	}
	if allZeros {
		t.Error("Embedding should not be all zeros")
	}

	// Verify embedding contains reasonable values (not NaN or Inf)
	for i, val := range embedding {
		if math.IsNaN(float64(val)) {
			t.Errorf("Embedding contains NaN at index %d", i)
		}
		if math.IsInf(float64(val), 0) {
			t.Errorf("Embedding contains Inf at index %d", i)
		}
	}
}

func TestBatchEmbedding(t *testing.T) {
	model, err := all_minilm_l6_v2.NewModel()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	sentences := []string{
		"Hello, world! This is a test sentence.",
		"This is another different sentence.",
		"A third sentence with different content.",
	}

	embeddings, err := model.ComputeBatch(sentences)
	if err != nil {
		t.Fatalf("Failed to compute batch embeddings: %v", err)
	}

	// Check batch size
	if len(embeddings) != len(sentences) {
		t.Errorf("Expected %d embeddings, got %d", len(sentences), len(embeddings))
	}

	// Check dimensions for each embedding
	expectedDim := 384
	for i, embedding := range embeddings {
		if len(embedding) != expectedDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, expectedDim, len(embedding))
		}

		// Verify embedding is not all zeros
		allZeros := true
		for _, val := range embedding {
			if val != 0.0 {
				allZeros = false
				break
			}
		}
		if allZeros {
			t.Errorf("Embedding %d should not be all zeros", i)
		}

		// Verify embedding contains reasonable values
		for j, val := range embedding {
			if math.IsNaN(float64(val)) {
				t.Errorf("Embedding %d contains NaN at index %d", i, j)
			}
			if math.IsInf(float64(val), 0) {
				t.Errorf("Embedding %d contains Inf at index %d", i, j)
			}
		}
	}

	// Verify that different sentences produce different embeddings
	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			if vectorsEqual(embeddings[i], embeddings[j]) {
				t.Errorf("Embeddings %d and %d should be different but are identical", i, j)
			}
		}
	}
}

func TestConsistentEmbeddingForSameSentence(t *testing.T) {
	model, err := all_minilm_l6_v2.NewModel()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	// Test with same sentence appearing twice in batch, plus a different sentence
	sentences := []string{
		"This is a repeated sentence.",
		"This is a different sentence for variety.",
		"This is a repeated sentence.", // Same as first
	}

	embeddings, err := model.ComputeBatch(sentences)
	if err != nil {
		t.Fatalf("Failed to compute batch embeddings: %v", err)
	}

	// Check that embeddings for identical sentences (index 0 and 2) are the same
	if !vectorsEqual(embeddings[0], embeddings[2]) {
		t.Error("Embeddings for identical sentences should be the same")
	}

	// Check that the different sentence (index 1) produces a different embedding
	if vectorsEqual(embeddings[0], embeddings[1]) {
		t.Error("Different sentences should produce different embeddings")
	}
	if vectorsEqual(embeddings[1], embeddings[2]) {
		t.Error("Different sentences should produce different embeddings")
	}

	// Verify dimensions are correct
	expectedDim := 384
	for i, embedding := range embeddings {
		if len(embedding) != expectedDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, expectedDim, len(embedding))
		}
	}

	// Verify embeddings are not all zeros
	for i, embedding := range embeddings {
		allZeros := true
		for _, val := range embedding {
			if val != 0.0 {
				allZeros = false
				break
			}
		}
		if allZeros {
			t.Errorf("Embedding %d should not be all zeros", i)
		}
	}
}

func TestSingleVsBatchConsistency(t *testing.T) {
	model, err := all_minilm_l6_v2.NewModel()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	sentence := "Testing consistency between single and batch computation."

	// Compute single embedding
	singleEmbedding, err := model.Compute(sentence)
	if err != nil {
		t.Fatalf("Failed to compute single embedding: %v", err)
	}

	// Compute batch embedding with just one sentence
	batchEmbeddings, err := model.ComputeBatch([]string{sentence})
	if err != nil {
		t.Fatalf("Failed to compute batch embedding: %v", err)
	}

	if len(batchEmbeddings) != 1 {
		t.Fatalf("Expected 1 embedding from batch, got %d", len(batchEmbeddings))
	}

	// Compare single vs batch result
	if !vectorsEqual(singleEmbedding, batchEmbeddings[0]) {
		t.Error("Single embedding should be identical to batch embedding for the same sentence")
	}
}

func TestEmptyBatch(t *testing.T) {
	model, err := all_minilm_l6_v2.NewModel()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	// Test empty batch
	embeddings, err := model.ComputeBatch([]string{})
	if err != nil {
		t.Fatalf("Failed to compute empty batch: %v", err)
	}

	if embeddings != nil {
		t.Error("Expected nil embeddings for empty batch")
	}
}

// Helper function to compare two vectors for equality with a small tolerance
func vectorsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}

	const tolerance = 1e-6
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tolerance {
			return false
		}
	}
	return true
}
