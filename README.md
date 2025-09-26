# all-MiniLM-L6-v2-go 

[![Build and Test](https://github.com/clems4ever/all-minilm-l6-v2-go/actions/workflows/build-and-release.yml/badge.svg)](https://github.com/clems4ever/all-minilm-l6-v2-go/actions/workflows/build-and-release.yml)

A Go implementation of the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence transformer model for generating high-quality sentence embeddings.

This library provides a native Go interface to compute sentence embeddings using the all-MiniLM-L6-v2 model, which is optimized for semantic similarity tasks and runs efficiently on CPU.

## Features

- **High Performance**: Optimized for both single sentence and batch processing
- **Easy Integration**: Simple Go API with minimal dependencies
- **Embedded Model**: Model weights and tokenizer are embedded in the binary
- **Well Tested**: Comprehensive test suite ensuring reliability
- **384-dimensional embeddings**: Compatible with the original model output

## Quick Start

### Docker (No Installation Required)

Try it out immediately with Docker:

```bash
echo "hello" | docker run -i ghcr.io/clems4ever/all-minilm-l6-v2-go:main . -o json
```

### Go Library

Install the library:

```bash
go get github.com/clems4ever/all-minilm-l6-v2-go
```

Use it in your code:

```go
package main

import (
    "fmt"
    "log"

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

	// computeCosineSimilarity..
	// ...
}
```

## Installation

### Prerequisites

This library requires ONNX Runtime to be installed on your system:

**Ubuntu/Debian:**
```bash
ORT_VERSION=1.23.0
wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz
tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz
sudo cp -r onnxruntime-linux-x64-${ORT_VERSION}/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-${ORT_VERSION}/lib/* /usr/local/lib/
```

**Other systems:** Please refer to the [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/).

## Performance Tips

1. **Use batch processing** when computing embeddings for multiple sentences - it's significantly more efficient than individual calls.

2. **Reuse model instances** - creating a model is expensive, so create once and reuse.

3. **Proper cleanup** - always call `Close()` to free resources when done.

4. **Future optimization**: Performance could be further improved by pre-allocating and reusing input and output tensors instead of dynamically allocating them for each batch. Currently, tensors are created and destroyed for every `ComputeBatch()` call, which adds allocation overhead.

## Testing

Run the test suite to ensure everything is working correctly:

```bash
ONNXRUNTIME_LIB_PATH=libonnxruntime.so go test ./all_minilm_l6_v2 -v
```

## Model Information

- **Model**: all-MiniLM-L6-v2
- **Embedding Dimension**: 384
- **Max Sequence Length**: Dynamic (handled automatically)
- **Model Size**: ~90MB (embedded)
- **Performance**: Optimized for CPU inference

This model was trained on a large corpus of sentence pairs and is designed to capture semantic meaning effectively. It's particularly good for:
- Semantic search
- Text clustering
- Similarity matching
- Information retrieval tasks

## Dependencies

- [ONNX Runtime Go](https://github.com/yalue/onnxruntime_go) - ONNX Runtime bindings for Go
- [Tokenizer](https://github.com/sugarme/tokenizer) - HuggingFace tokenizer implementation in Go (custom fork)

## FAQ

### Q: I'm getting "failed to initialize onnx runtime: Platform-specific initialization failed: Error loading ONNX shared library"

**Error message:**
```
failed to initialize onnx runtime: Platform-specific initialization failed: Error loading ONNX shared library "onnxruntime.so": onnxruntime.so: cannot open shared object file: No such file or directory
```

**Solution:**
This error occurs when the ONNX Runtime shared library cannot be found. Here are the solutions:

1. **Set the environment variable** (Quick fix):
   ```bash
   export ONNXRUNTIME_LIB_PATH=libonnxruntime.so
   go run your_program.go
   ```
2. **For testing**:
   ```bash
   ONNXRUNTIME_LIB_PATH=libonnxruntime.so go test ./all_minilm_l6_v2 -v
   ```

3. **Specify the runtime path in code**:
   ```go
   model, err := all_minilm_l6_v2.NewModel(
       all_minilm_l6_v2.WithRuntimePath("libonnxruntime.so"))
   ```

4. **Make sure ONNX Runtime is properly installed** following the installation instructions above.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is available under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Original model by [Sentence Transformers](https://www.sbert.net/)
- HuggingFace model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)