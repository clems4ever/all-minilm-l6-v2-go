package all_minilm_l6_v2_test

import (
	"fmt"
	"testing"

	"github.com/clems4ever/all-minilm-l6-v2-go/all_minilm_l6_v2"
)

func newModel(t *testing.B) *all_minilm_l6_v2.Model {
	benchModel, err := all_minilm_l6_v2.NewModel()
	if err != nil {
		panic(fmt.Sprintf("Failed to create benchmark model: %v", err))
	}
	t.Cleanup(func() {
		benchModel.Close()
	})
	return benchModel
}

// BenchmarkSingleSentence benchmarks single sentence processing
func BenchmarkSingleSentence(b *testing.B) {
	sentence := "This is a test sentence for benchmarking single sentence processing performance."

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.Compute(sentence)
		if err != nil {
			b.Fatalf("Failed to compute embedding: %v", err)
		}
	}
}

// BenchmarkSingleSentenceShort benchmarks short sentence processing
func BenchmarkSingleSentenceShort(b *testing.B) {
	sentence := "Short test."

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.Compute(sentence)
		if err != nil {
			b.Fatalf("Failed to compute embedding: %v", err)
		}
	}
}

// BenchmarkSingleSentenceLong benchmarks long sentence processing
func BenchmarkSingleSentenceLong(b *testing.B) {
	sentence := "This is a very long test sentence for benchmarking purposes that contains many words and should test the performance of the model with longer input sequences that might be more common in real-world applications where users provide detailed text descriptions or longer documents that need to be processed efficiently."

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.Compute(sentence)
		if err != nil {
			b.Fatalf("Failed to compute embedding: %v", err)
		}
	}
}

// BenchmarkBatch2 benchmarks batch processing with 2 sentences
func BenchmarkBatch2(b *testing.B) {
	sentences := []string{
		"This is the first test sentence for batch processing.",
		"This is the second test sentence for batch processing.",
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.ComputeBatch(sentences)
		if err != nil {
			b.Fatalf("Failed to compute batch embeddings: %v", err)
		}
	}
}

// BenchmarkBatch4 benchmarks batch processing with 4 sentences
func BenchmarkBatch4(b *testing.B) {
	sentences := []string{
		"First test sentence for batch processing performance measurement.",
		"Second test sentence with different content for variety.",
		"Third sentence to test batch processing capabilities.",
		"Fourth and final sentence in this benchmark test batch.",
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.ComputeBatch(sentences)
		if err != nil {
			b.Fatalf("Failed to compute batch embeddings: %v", err)
		}
	}
}

// BenchmarkBatch8 benchmarks batch processing with 8 sentences
func BenchmarkBatch8(b *testing.B) {
	sentences := []string{
		"First test sentence for batch processing performance measurement.",
		"Second test sentence with different content for variety.",
		"Third sentence to test batch processing capabilities.",
		"Fourth sentence in this benchmark test batch.",
		"Fifth sentence continuing the batch processing test.",
		"Sixth sentence with more content for comprehensive testing.",
		"Seventh sentence approaching the end of this batch.",
		"Eighth and final sentence in this larger benchmark batch.",
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.ComputeBatch(sentences)
		if err != nil {
			b.Fatalf("Failed to compute batch embeddings: %v", err)
		}
	}
}

// BenchmarkBatch16 benchmarks batch processing with 16 sentences
func BenchmarkBatch16(b *testing.B) {
	sentences := make([]string, 16)
	for i := range sentences {
		sentences[i] = fmt.Sprintf("Test sentence number %d for batch processing performance measurement and evaluation.", i+1)
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.ComputeBatch(sentences)
		if err != nil {
			b.Fatalf("Failed to compute batch embeddings: %v", err)
		}
	}
}

// BenchmarkBatch32 benchmarks batch processing with 32 sentences
func BenchmarkBatch32(b *testing.B) {
	sentences := make([]string, 32)
	for i := range sentences {
		sentences[i] = fmt.Sprintf("Test sentence number %d for batch processing performance measurement and evaluation with more detailed content.", i+1)
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.ComputeBatch(sentences)
		if err != nil {
			b.Fatalf("Failed to compute batch embeddings: %v", err)
		}
	}
}

// BenchmarkVsSingle4Individual compares batch processing vs individual calls
func BenchmarkVsSingle4Individual(b *testing.B) {
	sentences := []string{
		"First test sentence for individual processing comparison.",
		"Second test sentence with different content.",
		"Third sentence to test individual processing.",
		"Fourth sentence in this comparison test.",
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		for _, sentence := range sentences {
			_, err := benchModel.Compute(sentence)
			if err != nil {
				b.Fatalf("Failed to compute embedding: %v", err)
			}
		}
	}
}

// BenchmarkModelCreation benchmarks model initialization (expensive operation)
// Note: This benchmark is commented out due to ONNX runtime single initialization limitation
// Uncomment and run separately to measure model creation performance
/*
func BenchmarkModelCreation(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		model, err := all_minilm_l6_v2.newModel(b)
		if err != nil {
			b.Fatalf("Failed to create model: %v", err)
		}
		err = model.Close()
		if err != nil {
			b.Fatalf("Failed to close model: %v", err)
		}
	}
}
*/

// BenchmarkVariableLengthBatch benchmarks batch with variable length sentences
func BenchmarkVariableLengthBatch(b *testing.B) {
	sentences := []string{
		"Short.",
		"This is a medium length sentence for testing.",
		"This is a much longer sentence that contains significantly more words and content to test how the model handles variable length inputs in a batch processing scenario.",
		"Medium length test sentence.",
		"Another short one.",
		"This sentence has a moderate amount of content for testing purposes and evaluation of the model performance.",
	}

	benchModel := newModel(b)
	b.ReportAllocs()

	for b.Loop() {
		_, err := benchModel.ComputeBatch(sentences)
		if err != nil {
			b.Fatalf("Failed to compute batch embeddings: %v", err)
		}
	}
}
