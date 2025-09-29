package all_minilm_l6_v2

import (
	"bytes"
	"fmt"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

type Tokenizer struct {
	tk *tokenizer.Tokenizer
}

func NewTokenizer() (*Tokenizer, error) {
	tk, err := pretrained.FromReader(bytes.NewBuffer(embeddedTokenizer))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}
	return &Tokenizer{
		tk: tk,
	}, nil
}

func (tk *Tokenizer) Encode(s string, addSpecialTokens bool) (*tokenizer.Encoding, error) {
	return tk.tk.EncodeSingle(s, addSpecialTokens)
}

func (tk *Tokenizer) Decode(ids []int, skipSpecialTokens bool) string {
	return tk.tk.Decode(ids, skipSpecialTokens)
}
