package all_minilm_l6_v2

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed tokenizer.json
var embeddedTokenizer []byte

//go:embed model.onnx
var onnxModel []byte

type Model struct {
	tk      tokenizer.Tokenizer
	session *ort.DynamicAdvancedSession

	runtimePath string
}

type ModelOption = func(*Model)

func WithRuntimePath(path string) ModelOption {
	return func(m *Model) {
		m.runtimePath = path
	}
}

func NewModel(opts ...ModelOption) (*Model, error) {
	model := new(Model)

	for _, opt := range opts {
		opt(model)
	}

	tk, err := pretrained.FromReader(bytes.NewBuffer(embeddedTokenizer))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	if model.runtimePath != "" {
		ort.SetSharedLibraryPath(model.runtimePath)
	} else {
		path, ok := os.LookupEnv("ONNXRUNTIME_LIB_PATH")
		if ok {
			ort.SetSharedLibraryPath(path)
		}
	}

	err = ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize onnx runtime: %w", err)
	}

	// Create a dynamic session that accepts tensors at runtime
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"sentence_embedding"}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(onnxModel, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Model{
		tk:      *tk,
		session: session,
	}, nil
}

func (m *Model) Close() error {
	if m.session != nil {
		m.session.Destroy()
	}
	err := ort.DestroyEnvironment()
	return err
}

func (m *Model) Compute(sentence string) ([]float32, error) {
	results, err := m.ComputeBatch([]string{sentence})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

func (m *Model) ComputeBatch(sentences []string) ([][]float32, error) {
	if len(sentences) == 0 {
		return nil, nil
	}

	inputBatch := []tokenizer.EncodeInput{}
	for _, s := range sentences {
		inputBatch = append(inputBatch, tokenizer.NewSingleEncodeInput(tokenizer.NewRawInputSequence(s)))
	}
	encodings, err := m.tk.EncodeBatch(inputBatch, true)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize sentence: %w", err)
	}

	batchSize := len(sentences)
	seqLength := len(encodings[0].Ids)
	hiddenSize := 384

	inputShape := ort.NewShape(int64(batchSize), int64(seqLength))

	// Create input tensors dynamically based on the actual sequence length
	inputIdsData := make([]int64, batchSize*seqLength)
	attentionMaskData := make([]int64, batchSize*seqLength)
	tokenTypeIdsData := make([]int64, batchSize*seqLength)

	for b := range batchSize {
		for i, id := range encodings[b].Ids {
			inputIdsData[b*seqLength+i] = int64(id)
		}
		for i, mask := range encodings[b].AttentionMask {
			attentionMaskData[b*seqLength+i] = int64(mask)
		}
		for i, typeId := range encodings[b].TypeIds {
			tokenTypeIdsData[b*seqLength+i] = int64(typeId)
		}
	}

	inputIdsTensor, err := ort.NewTensor(inputShape, inputIdsData)
	if err != nil {
		return nil, fmt.Errorf("failed creating input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMaskData)
	if err != nil {
		return nil, fmt.Errorf("failed creating attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	tokenTypeIdsTensor, err := ort.NewTensor(inputShape, tokenTypeIdsData)
	if err != nil {
		return nil, fmt.Errorf("failed creating token_type_ids tensor: %w", err)
	}
	defer tokenTypeIdsTensor.Destroy()

	sentenceOutputShape := ort.NewShape(int64(batchSize), int64(hiddenSize))
	sentenceOutputTensor, err := ort.NewEmptyTensor[float32](sentenceOutputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create empty tensor: %w", err)
	}
	defer sentenceOutputTensor.Destroy()

	inputTensors := []ort.Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor}
	outputTensors := []ort.Value{sentenceOutputTensor}

	err = m.session.Run(inputTensors, outputTensors)
	if err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	flatOutput := sentenceOutputTensor.GetData()

	// We want to have the highest throughput possible so we skip this assuming that the size of the
	// array is correct, that is (batshSize, hiddenSize).
	//
	expectedTotalSize := batchSize * hiddenSize
	if len(flatOutput) != expectedTotalSize {
		return nil, fmt.Errorf("unexpected output tensor size: got %d elements, expected %d elements", len(flatOutput), expectedTotalSize)
	}

	results := make([][]float32, batchSize)
	for i := range batchSize {
		start := i * hiddenSize
		end := start + hiddenSize
		results[i] = make([]float32, hiddenSize)
		copy(results[i], flatOutput[start:end])
	}

	return results, nil
}
