package main

import (
	"fmt"

	"github.com/clems4ever/all-minilm-l6-v2-go/all_minilm_l6_v2"
)

func main() {
	model, err := all_minilm_l6_v2.NewModel(
		all_minilm_l6_v2.WithRuntimePath("libonnxruntime.so"))
	if err != nil {
		panic(err)
	}
	defer model.Close()

	// Base sentence to compare against
	baseSentence := "The dog is running in the park"

	// Three candidate sentences with varying degrees of similarity
	candidates := []string{
		"A dog runs through the park",      // Very similar
		"The cat is sleeping on the couch", // Somewhat similar
		"I love eating pizza for dinner",   // Not similar
	}

	// Compute embeddings
	baseEmbedding, _ := model.Compute(baseSentence)
	candidateEmbeddings, _ := model.ComputeBatch(candidates)

	displaySimilarities(baseSentence, candidates, baseEmbedding, candidateEmbeddings)
}

func displaySimilarities(
	baseSentence string,
	candidates []string,
	baseEmbedding []float32,
	candidateEmbeddings [][]float32,
) {
	// Display results in a table
	fmt.Printf("Base: %s\n\n", baseSentence)
	fmt.Println("Similarity | Sentence")
	fmt.Println("-----------|---------")

	for i, candidate := range candidates {
		similarity := all_minilm_l6_v2.CosineSimilarity(baseEmbedding, candidateEmbeddings[i])
		fmt.Printf("   %.4f   | %s\n", similarity, candidate)
	}
}
