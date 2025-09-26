# Performance Benchmarks

Performance benchmark results for the all-minilm-l6-v2-go library.

## Test Environment

- **CPU**: 13th Gen Intel(R) Core(TM) i9-13900K
- **OS**: Linux (amd64)
- **Go Version**: 1.25.1

## Single Sentence Processing

| Sentence Type | Operations | ns/op | Allocs/op | B/op |
|---------------|------------|-------|-----------|------|
| Short sentence ("Short test.") | 266 | 4,473,787 | 797 | 72,128 |
| Medium sentence (77 chars) | 255 | 4,605,916 | 2,354 | 206,857 |
| Long sentence (264 chars) | 253 | 4,725,359 | 7,663 | 533,261 |

## Batch Processing

| Batch Size | Operations | ns/op | Allocs/op | B/op |
|------------|------------|-------|-----------|------|
| 2 sentences | 165 | 7,238,155 | 3,578 | 298,256 |
| 4 sentences | 88 | 12,671,776 | 7,261 | 600,790 |
| 8 sentences | 44 | 23,963,903 | 14,074 | 1,159,160 |
| 16 sentences | 22 | 46,594,877 | 37,774 | 3,602,897 |
| 32 sentences | 12 | 98,840,400 | 94,228 | 7,313,192 |

## Batch vs Individual Processing (4 sentences)

| Method | Operations | ns/op | Allocs/op | B/op |
|--------|------------|-------|-----------|------|
| 4 individual calls | 62 | 17,985,727 | 6,322 | 520,739 |
| 1 batch of 4 | 88 | 12,671,776 | 7,261 | 600,790 |

## Variable Length Batch (6 sentences)

| Test Type | Operations | ns/op | Allocs/op | B/op |
|-----------|------------|-------|-----------|------|
| Mixed sentence lengths | 64 | 18,289,558 | 11,548 | 936,625 |

## Running Benchmarks

```bash
# Run all benchmarks
go test ./all_minilm_l6_v2 -bench=. -run=^$ -benchmem

# Run specific benchmark
go test ./all_minilm_l6_v2 -bench=BenchmarkBatch8 -run=^$ -benchmem

# Run with multiple iterations
go test ./all_minilm_l6_v2 -bench=. -run=^$ -benchmem -count=3

# Generate CPU profile
go test ./all_minilm_l6_v2 -bench=BenchmarkBatch8 -run=^$ -cpuprofile=cpu.prof
```

## Available Benchmarks

- `BenchmarkSingleSentence` - Standard sentence processing
- `BenchmarkSingleSentenceShort` - Short sentence processing  
- `BenchmarkSingleSentenceLong` - Long sentence processing
- `BenchmarkBatch2` - Batch of 2 sentences
- `BenchmarkBatch4` - Batch of 4 sentences
- `BenchmarkBatch8` - Batch of 8 sentences
- `BenchmarkBatch16` - Batch of 16 sentences
- `BenchmarkBatch32` - Batch of 32 sentences
- `BenchmarkVsSingle4Individual` - 4 individual calls for comparison
- `BenchmarkVariableLengthBatch` - Mixed sentence lengths in batch