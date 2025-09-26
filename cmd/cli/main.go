package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/clems4ever/all-minilm-l6-v2-go/all_minilm_l6_v2"
	"github.com/spf13/cobra"
)

var (
	runtimePath  string
	outputFormat string
	batchMode    bool
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "all-minilm-l6-v2-go",
		Short: "Generate sentence embeddings using all-MiniLM-L6-v2 model",
		Long:  `A CLI tool to generate 384-dimensional sentence embeddings from text input using the all-MiniLM-L6-v2 model.`,
		Run:   runEmbedding,
	}

	rootCmd.Flags().StringVar(&runtimePath, "runtime-path", "", "Path to ONNX Runtime shared library (default: use ONNXRUNTIME_LIB_PATH env var)")
	rootCmd.Flags().StringVarP(&outputFormat, "output", "o", "values", "Output format: 'values' (space-separated), 'json', or 'json-pretty'")
	rootCmd.Flags().BoolVarP(&batchMode, "batch", "b", false, "Process multiple lines as a batch (more efficient for multiple sentences)")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runEmbedding(cmd *cobra.Command, args []string) {
	var opts []all_minilm_l6_v2.ModelOption
	if runtimePath != "" {
		opts = append(opts, all_minilm_l6_v2.WithRuntimePath(runtimePath))
	}

	model, err := all_minilm_l6_v2.NewModel(opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	// Read input from stdin
	reader := bufio.NewReader(os.Stdin)
	var sentences []string

	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			if line != "" {
				sentences = append(sentences, strings.TrimSpace(line))
			}
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
			os.Exit(1)
		}

		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			sentences = append(sentences, trimmed)
		}
	}

	if len(sentences) == 0 {
		fmt.Fprintf(os.Stderr, "No input provided\n")
		os.Exit(1)
	}

	// Compute embeddings
	if batchMode && len(sentences) > 1 {
		embeddings, err := model.ComputeBatch(sentences)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to compute batch embeddings: %v\n", err)
			os.Exit(1)
		}

		for i, embedding := range embeddings {
			outputEmbedding(sentences[i], embedding)
		}
	} else {
		// Process individually
		for _, sentence := range sentences {
			embedding, err := model.Compute(sentence)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Failed to compute embedding for '%s': %v\n", sentence, err)
				os.Exit(1)
			}
			outputEmbedding(sentence, embedding)
		}
	}
}

func outputEmbedding(sentence string, embedding []float32) {
	switch outputFormat {
	case "json":
		result := map[string]interface{}{
			"sentence":  sentence,
			"embedding": embedding,
		}
		jsonData, _ := json.Marshal(result)
		fmt.Println(string(jsonData))

	case "json-pretty":
		result := map[string]interface{}{
			"sentence":  sentence,
			"embedding": embedding,
		}
		jsonData, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(jsonData))

	case "values":
		fallthrough
	default:
		fmt.Print("# " + sentence + "\n")
		for i, val := range embedding {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf("%.6f", val)
		}
		fmt.Println()
	}
}
