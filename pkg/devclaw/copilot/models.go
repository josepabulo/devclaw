// Package copilot – models.go is the single source of truth for per-model
// behaviour: context window, output cap, sampling support, tokenizer ratio and
// pricing. getModelDefaults, getModelContextWindowByName, charsPerToken and the
// usage tracker resolve through LookupModel; each keeps a family-heuristic
// fallback for custom or fine-tuned names the table cannot know about.
package copilot

import (
	"sort"
	"strings"
)

// ModelSpec holds everything DevClaw needs to know about one model.
// A zero EffortLevels means the model takes no reasoning_effort.
type ModelSpec struct {
	Canonical     string
	ContextWindow int
	// BetaContextWindow is the window reachable through an opt-in beta header,
	// zero when the model has none. The Claude 4.x family reaches 1M this way.
	BetaContextWindow       int
	MaxOutput               int
	SupportsTemperature     bool
	DefaultTemperature      float64
	UsesMaxCompletionTokens bool
	SupportsTools           bool
	// AcceptsReasoningEffort says the model takes the reasoning_effort param at
	// all; EffortLevels lists the accepted values when they are known.
	AcceptsReasoningEffort bool
	EffortLevels           []string
	CharsPerToken          float64
	InputPer1M             float64
	OutputPer1M            float64
}

// foldModelName canonicalises a model name for matching: lowercase, and "." to
// "-" so the dotted spellings (claude-opus-4.6, used by the pricing table) and
// the dashed API ids (claude-opus-4-6) collapse to the same key.
func foldModelName(name string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(name)), ".", "-")
}

// Reasoning effort levels, per the OpenAI model docs.
var (
	effortGPT6 = []string{"medium", "high", "xhigh", "max"}
	effortGLM  = []string{"low", "high", "max"}
	effortGPT5 = []string{"none", "low", "medium", "high", "xhigh", "max"}
)

// The 1M window on the Claude 4.x family is opt-in through the
// context-1m-2025-08-07 beta header (see isAnthropic1MModel), so those entries
// keep the ungated 200k. MaxOutput is what DevClaw requests by default, not the
// model's ceiling: the 128k caps need the streaming path to avoid HTTP timeouts.
// modelRegistry is keyed by canonical name; entries double as prefix rules, so
// "gpt-5" also covers "gpt-5-2025-08-07". Longest prefix wins.
var modelRegistry = []ModelSpec{
	// ── OpenAI ──
	{Canonical: "gpt-4o", ContextWindow: 128000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.7, InputPer1M: 2.50, OutputPer1M: 10.00},
	{Canonical: "gpt-4o-mini", ContextWindow: 128000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.7, InputPer1M: 0.15, OutputPer1M: 0.60},
	{Canonical: "gpt-4.5", ContextWindow: 8192, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.7},
	{Canonical: "gpt-4.5-preview", ContextWindow: 8192, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.7, InputPer1M: 75.00, OutputPer1M: 150.00},
	{Canonical: "gpt-4.1", ContextWindow: 8192, MaxOutput: 16384, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, CharsPerToken: 3.7},
	{Canonical: "gpt-4-turbo", ContextWindow: 128000, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.7},
	{Canonical: "gpt-4", ContextWindow: 8192, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.7},
	{Canonical: "gpt-5", ContextWindow: 128000, MaxOutput: 16384, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, CharsPerToken: 3.7, InputPer1M: 2.00, OutputPer1M: 8.00},
	{Canonical: "gpt-5-mini", ContextWindow: 128000, MaxOutput: 16384, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, CharsPerToken: 3.7, InputPer1M: 0.15, OutputPer1M: 0.60},
	{Canonical: "gpt-5.6", ContextWindow: 1050000, MaxOutput: 16384, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, EffortLevels: effortGPT5, CharsPerToken: 3.7},
	{Canonical: "gpt-6-astra", ContextWindow: 1050000, MaxOutput: 16384, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, EffortLevels: effortGPT6, CharsPerToken: 3.7, InputPer1M: 10.00, OutputPer1M: 50.00},
	{Canonical: "o1", ContextWindow: 128000, MaxOutput: 100000, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, CharsPerToken: 4.0},
	{Canonical: "o3", ContextWindow: 128000, MaxOutput: 100000, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, CharsPerToken: 4.0},
	{Canonical: "o4", ContextWindow: 128000, MaxOutput: 100000, DefaultTemperature: 0.7, UsesMaxCompletionTokens: true, SupportsTools: true, AcceptsReasoningEffort: true, CharsPerToken: 4.0},

	// ── Anthropic ──
	{Canonical: "claude-opus-4", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "claude-opus-4.5", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5, InputPer1M: 5.00, OutputPer1M: 25.00},
	{Canonical: "claude-opus-4.6", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5, InputPer1M: 5.00, OutputPer1M: 25.00},
	{Canonical: "claude-sonnet-4", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "claude-sonnet-4.5", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5, InputPer1M: 3.00, OutputPer1M: 15.00},
	{Canonical: "claude-sonnet-4.6", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5, InputPer1M: 3.00, OutputPer1M: 15.00},
	{Canonical: "claude-haiku", ContextWindow: 128000, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "claude-opus-4-7", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.0, InputPer1M: 5.00, OutputPer1M: 25.00},
	{Canonical: "claude-opus-4-8", ContextWindow: 200000, BetaContextWindow: 1000000, MaxOutput: 16384, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.0, InputPer1M: 5.00, OutputPer1M: 25.00},
	{Canonical: "claude-opus-5", ContextWindow: 1000000, MaxOutput: 16384, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.0, InputPer1M: 5.00, OutputPer1M: 25.00},
	{Canonical: "claude-sonnet-5", ContextWindow: 1000000, MaxOutput: 16384, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.0, InputPer1M: 2.00, OutputPer1M: 10.00},
	{Canonical: "claude-fable-5", ContextWindow: 1000000, MaxOutput: 16384, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.0, InputPer1M: 10.00, OutputPer1M: 50.00},
	{Canonical: "claude-fable-5-1", ContextWindow: 1000000, MaxOutput: 16384, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.0, InputPer1M: 10.00, OutputPer1M: 50.00},
	{Canonical: "claude-haiku-4-5", ContextWindow: 200000, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5, InputPer1M: 1.00, OutputPer1M: 5.00},
	{Canonical: "claude-3", ContextWindow: 200000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "claude-3.5-sonnet", ContextWindow: 200000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5, InputPer1M: 3.00, OutputPer1M: 15.00},

	// ── Google ──
	{Canonical: "gemini-1.5", ContextWindow: 128000, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "gemini-2", ContextWindow: 128000, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "gemini-2.5", ContextWindow: 128000, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "gemini-3", ContextWindow: 128000, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 1.0, SupportsTools: true, CharsPerToken: 3.5},

	// ── Z.AI (GLM) ──
	{Canonical: "glm-4", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5},
	{Canonical: "glm-4.7", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 0.50, OutputPer1M: 1.50},
	{Canonical: "glm-4.7-flash", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 0.10, OutputPer1M: 0.40},
	{Canonical: "glm-4.7-flashx", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 0.10, OutputPer1M: 0.40},
	{Canonical: "glm-5", ContextWindow: 202752, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 1.00, OutputPer1M: 3.20},
	{Canonical: "glm-5-code", ContextWindow: 202752, MaxOutput: 8192, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 1.20, OutputPer1M: 5.00},
	{Canonical: "glm-5-turbo", ContextWindow: 202752, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 0.80, OutputPer1M: 2.50},
	{Canonical: "glm-5.2", ContextWindow: 1048576, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5, InputPer1M: 1.40, OutputPer1M: 4.40},
	{Canonical: "glm-5.3", ContextWindow: 1048576, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, AcceptsReasoningEffort: true, EffortLevels: effortGLM, CharsPerToken: 2.5, InputPer1M: 1.40, OutputPer1M: 4.40},
	{Canonical: "glm-5.3-flash", ContextWindow: 1048576, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, AcceptsReasoningEffort: true, EffortLevels: effortGLM, CharsPerToken: 2.5, InputPer1M: 0.15, OutputPer1M: 0.50},
	{Canonical: "glm-5.1", ContextWindow: 1048576, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5},

	// ── xAI ──
	{Canonical: "grok", ContextWindow: 128000, MaxOutput: 16384, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 4.0},

	// ── Local / open-weight ──
	{Canonical: "llama", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "codellama", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "mistral", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 3.5},
	{Canonical: "qwen", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5},
	{Canonical: "gemma", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 4.0},
	{Canonical: "phi", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 4.0},
	{Canonical: "deepseek", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 2.5},
	{Canonical: "command-r", ContextWindow: 128000, MaxOutput: 4096, SupportsTemperature: true, DefaultTemperature: 0.7, SupportsTools: true, CharsPerToken: 4.0},
}

var (
	modelByName   map[string]ModelSpec
	modelPrefixes []ModelSpec // longest folded Canonical first
)

func init() {
	modelByName = make(map[string]ModelSpec, len(modelRegistry))
	modelPrefixes = make([]ModelSpec, len(modelRegistry))
	copy(modelPrefixes, modelRegistry)

	for _, m := range modelRegistry {
		modelByName[foldModelName(m.Canonical)] = m
	}
	sort.SliceStable(modelPrefixes, func(i, j int) bool {
		return len(foldModelName(modelPrefixes[i].Canonical)) > len(foldModelName(modelPrefixes[j].Canonical))
	})
}

// LookupModel resolves a model name to its spec. Resolution order is strict:
// exact match, then the segment after the last "/" (gateway and OpenRouter
// prefixes such as "gatorllm/gpt-5.4" or "anthropic/claude-opus-5"), then the
// longest matching prefix. The suffix step runs after the exact match, never
// before, so Ollama tags ("llama3:8b") and HuggingFace repos ("org/repo") are
// not mangled into a wrong match.
func LookupModel(name string) (ModelSpec, bool) {
	folded := foldModelName(name)
	if folded == "" {
		return ModelSpec{}, false
	}

	if m, ok := modelByName[folded]; ok {
		return m, true
	}

	suffix := folded
	if i := strings.LastIndex(folded, "/"); i >= 0 && i+1 < len(folded) {
		suffix = folded[i+1:]
		if m, ok := modelByName[suffix]; ok {
			return m, true
		}
	}

	for _, m := range modelPrefixes {
		key := foldModelName(m.Canonical)
		if prefixMatches(folded, key) || (suffix != folded && prefixMatches(suffix, key)) {
			return m, true
		}
	}
	return ModelSpec{}, false
}

// prefixMatches reports whether name starts with key on a segment boundary.
// A key ending in a digit must not swallow a longer number: folding turns
// "gpt-4.1" into the key "gpt-4-1", which would otherwise capture the real id
// "gpt-4-1106-preview" and send it the reasoning-model parameters.
func prefixMatches(name, key string) bool {
	if !strings.HasPrefix(name, key) {
		return false
	}
	if len(name) == len(key) {
		return true
	}
	last, next := key[len(key)-1], name[len(key)]
	return !isASCIIDigit(last) || !isASCIIDigit(next)
}

func isASCIIDigit(b byte) bool { return b >= '0' && b <= '9' }

// LookupModelPrice resolves pricing, skipping entries that carry none: a newer
// entry without a published price must not shadow the priced family prefix and
// silently report a zero cost.
func LookupModelPrice(name string) (in, out float64, ok bool) {
	if spec, found := LookupModel(name); found && (spec.InputPer1M > 0 || spec.OutputPer1M > 0) {
		return spec.InputPer1M, spec.OutputPer1M, true
	}

	folded := foldModelName(name)
	suffix := folded
	if i := strings.LastIndex(folded, "/"); i >= 0 && i+1 < len(folded) {
		suffix = folded[i+1:]
	}
	for _, m := range modelPrefixes {
		if m.InputPer1M == 0 && m.OutputPer1M == 0 {
			continue
		}
		key := foldModelName(m.Canonical)
		if prefixMatches(folded, key) || (suffix != folded && prefixMatches(suffix, key)) {
			return m.InputPer1M, m.OutputPer1M, true
		}
	}
	return 0, 0, false
}
