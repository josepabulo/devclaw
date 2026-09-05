package copilot

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// modelGolden pins the resolved behaviour of every model the registry knows.
// Values were captured from the pre-registry implementation, so the table is a
// change detector: a diff here means a deliberate decision, not a silent drift.
type modelGolden struct {
	name     string
	temp     bool
	dtemp    float64
	maxOut   int
	maxComp  bool
	tools    bool
	ctx      int
	charsTok float64
	// unresolved marks names that must NOT hit the registry and must reach the
	// family heuristic instead.
	unresolved bool
}

var modelGoldens = []modelGolden{
	{"gpt-4o", true, 0.7, 16384, false, true, 128000, 3.7, false},
	{"gpt-4o-mini", true, 0.7, 16384, false, true, 128000, 3.7, false},
	{"gpt-4o-2024-08-06", true, 0.7, 16384, false, true, 128000, 3.7, false},
	{"gpt-4.5-preview", true, 0.7, 16384, false, true, 8192, 3.7, false},
	{"gpt-4.1", false, 0.7, 16384, true, true, 8192, 3.7, false},
	{"gpt-4-turbo", true, 0.7, 0, false, true, 128000, 3.7, false},
	{"gpt-4", true, 0.7, 0, false, true, 8192, 3.7, false},
	{"gpt-5", false, 0.7, 16384, true, true, 128000, 3.7, false},
	{"gpt-5-mini", false, 0.7, 16384, true, true, 128000, 3.7, false},
	{"gpt-5.2", false, 0.7, 16384, true, true, 128000, 3.7, false},
	{"gpt-5.4", false, 0.7, 16384, true, true, 128000, 3.7, false},
	{"o1", false, 0.7, 100000, true, true, 128000, 4.0, false},
	{"o3-mini", false, 0.7, 100000, true, true, 128000, 4.0, false},
	{"o4", false, 0.7, 100000, true, true, 128000, 4.0, false},

	{"claude-opus-4", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-opus-4.5", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-opus-4.6", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-sonnet-4", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-sonnet-4-6", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-sonnet-4.6", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-sonnet-4.5", true, 1.0, 16384, false, true, 200000, 3.5, false},
	{"claude-haiku-4.5-20251001", true, 1.0, 8192, false, true, 200000, 3.5, false},
	{"claude-3", true, 1.0, 4096, false, true, 200000, 3.5, false},
	{"claude-3.5-sonnet", true, 1.0, 4096, false, true, 200000, 3.5, false},
	{"claude-3-opus", true, 1.0, 4096, false, true, 200000, 3.5, false},
	{"claude-3.7-sonnet", true, 1.0, 4096, false, true, 200000, 3.5, false},

	{"gemini-1.5-pro", true, 1.0, 8192, false, true, 128000, 3.5, false},
	{"gemini-2", true, 1.0, 8192, false, true, 128000, 3.5, false},
	{"gemini-2.5-pro", true, 1.0, 8192, false, true, 128000, 3.5, false},
	{"gemini-3", true, 1.0, 8192, false, true, 128000, 3.5, false},

	{"glm-4", true, 0.7, 4096, false, true, 128000, 2.5, false},
	{"glm-4.7", true, 0.7, 4096, false, true, 128000, 2.5, false},
	{"glm-4.7-flash", true, 0.7, 4096, false, true, 128000, 2.5, false},
	{"glm-5", true, 0.7, 8192, false, true, 202752, 2.5, false},
	{"glm-5-code", true, 0.7, 8192, false, true, 202752, 2.5, false},
	{"glm-5-turbo", true, 0.7, 16384, false, true, 202752, 2.5, false},
	{"glm-5.1", true, 0.7, 16384, false, true, 1048576, 2.5, false},

	{"grok-2", true, 0.7, 16384, false, true, 128000, 4.0, false},
	{"llama3", true, 0.7, 4096, false, true, 128000, 3.5, false},
	{"codellama", true, 0.7, 4096, false, true, 128000, 3.5, false},
	{"mistral-7b", true, 0.7, 4096, false, true, 128000, 3.5, false},
	{"qwen2", true, 0.7, 4096, false, true, 128000, 2.5, false},
	{"gemma2", true, 0.7, 4096, false, true, 128000, 4.0, false},
	{"phi3", true, 0.7, 4096, false, true, 128000, 4.0, false},
	{"deepseek-v3", true, 0.7, 4096, false, true, 128000, 2.5, false},
	{"command-r", true, 0.7, 4096, false, true, 128000, 4.0, false},

	// Legacy dated ids: folding "gpt-4.1" yields the key "gpt-4-1", which must
	// not swallow "gpt-4-1106-preview" and hand it reasoning-model parameters.
	{"gpt-4-1106-preview", true, 0.7, 0, false, true, 8192, 3.7, false},
	{"gpt-4-1106-vision-preview", true, 0.7, 0, false, true, 8192, 3.7, false},
	{"gpt-4-0613", true, 0.7, 0, false, true, 8192, 3.7, false},
	{"gpt-4-0125-preview", true, 0.7, 0, false, true, 8192, 3.7, false},

	// Bedrock-style ids separate the vendor with ".", which folding turns into
	// "-". These never reach the registry, so they pin the family fallback in
	// agent.go and prompt_layers.go as load-bearing.
	{"anthropic.claude-sonnet-4", true, 0.7, 0, false, true, 200000, 3.5, true},
	{"anthropic.claude-3-5-sonnet-20241022-v2:0", true, 0.7, 0, false, true, 200000, 3.5, true},

	// Unknown names must keep falling through to the conservative defaults.
	{"modelo-inexistente", true, 0.7, 0, false, true, 128000, 4.0, true},
	{"org/modelo-desconhecido", true, 0.7, 0, false, true, 128000, 4.0, true},
}

func TestModelGolden(t *testing.T) {
	for _, g := range modelGoldens {
		t.Run(g.name, func(t *testing.T) {
			d := getModelDefaults(g.name, "")
			if d.SupportsTemperature != g.temp {
				t.Errorf("SupportsTemperature = %v, want %v", d.SupportsTemperature, g.temp)
			}
			if d.DefaultTemperature != g.dtemp {
				t.Errorf("DefaultTemperature = %v, want %v", d.DefaultTemperature, g.dtemp)
			}
			if d.MaxOutputTokens != g.maxOut {
				t.Errorf("MaxOutputTokens = %d, want %d", d.MaxOutputTokens, g.maxOut)
			}
			if d.UsesMaxCompletionTokens != g.maxComp {
				t.Errorf("UsesMaxCompletionTokens = %v, want %v", d.UsesMaxCompletionTokens, g.maxComp)
			}
			if d.SupportsTools != g.tools {
				t.Errorf("SupportsTools = %v, want %v", d.SupportsTools, g.tools)
			}
			if got := getModelContextWindowByName(g.name); got != g.ctx {
				t.Errorf("contextWindow = %d, want %d", got, g.ctx)
			}
			if got := charsPerToken(g.name); got != g.charsTok {
				t.Errorf("charsPerToken = %v, want %v", got, g.charsTok)
			}
		})
	}
}

func TestFoldModelName(t *testing.T) {
	cases := map[string]string{
		"Claude-Opus-4.6":  "claude-opus-4-6",
		"claude-opus-4-6":  "claude-opus-4-6",
		"  GPT-5.6  ":      "gpt-5-6",
		"gatorllm/GPT-5.4": "gatorllm/gpt-5-4",
		"":                 "",
	}
	for in, want := range cases {
		if got := foldModelName(in); got != want {
			t.Errorf("foldModelName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestLookupModelResolution(t *testing.T) {
	t.Run("exact", func(t *testing.T) {
		m, ok := LookupModel("glm-5-turbo")
		if !ok || m.Canonical != "glm-5-turbo" {
			t.Fatalf("got %q ok=%v, want glm-5-turbo", m.Canonical, ok)
		}
	})

	t.Run("dotted and dashed spellings collapse", func(t *testing.T) {
		dotted, ok1 := LookupModel("claude-opus-4.6")
		dashed, ok2 := LookupModel("claude-opus-4-6")
		if !ok1 || !ok2 || dotted.Canonical != dashed.Canonical {
			t.Fatalf("dotted=%q(%v) dashed=%q(%v), want same spec", dotted.Canonical, ok1, dashed.Canonical, ok2)
		}
	})

	t.Run("gateway prefix resolves via suffix", func(t *testing.T) {
		bare, _ := LookupModel("gpt-5.4")
		prefixed, ok := LookupModel("gatorllm/gpt-5.4")
		if !ok || prefixed.Canonical != bare.Canonical {
			t.Fatalf("gatorllm/gpt-5.4 -> %q(%v), want %q", prefixed.Canonical, ok, bare.Canonical)
		}
	})

	t.Run("openrouter prefix resolves via suffix", func(t *testing.T) {
		m, ok := LookupModel("anthropic/claude-sonnet-4-6")
		if !ok || m.Canonical != "claude-sonnet-4.6" {
			t.Fatalf("got %q ok=%v, want claude-sonnet-4.6", m.Canonical, ok)
		}
	})

	t.Run("longest prefix wins", func(t *testing.T) {
		m, ok := LookupModel("gpt-4o-mini-2024-07-18")
		if !ok || m.Canonical != "gpt-4o-mini" {
			t.Fatalf("got %q ok=%v, want gpt-4o-mini", m.Canonical, ok)
		}
		if m.InputPer1M != 0.15 {
			t.Errorf("InputPer1M = %v, want 0.15 (must not fall back to gpt-4o pricing)", m.InputPer1M)
		}
	})

	t.Run("ollama tag is not mangled", func(t *testing.T) {
		m, ok := LookupModel("llama3:8b")
		if !ok || m.Canonical != "llama" {
			t.Fatalf("got %q ok=%v, want llama via prefix", m.Canonical, ok)
		}
	})

	t.Run("unknown huggingface repo does not resolve", func(t *testing.T) {
		if m, ok := LookupModel("org/modelo-desconhecido"); ok {
			t.Fatalf("resolved to %q, want no match", m.Canonical)
		}
	})

	t.Run("empty name does not resolve", func(t *testing.T) {
		if _, ok := LookupModel(""); ok {
			t.Fatal("empty name resolved, want no match")
		}
	})
}

// TestModelRegistryMatchesGolden validates the table itself against the same
// goldens, so a wrong entry cannot hide behind the legacy fallbacks.
func TestModelRegistryMatchesGolden(t *testing.T) {
	for _, g := range modelGoldens {
		spec, ok := LookupModel(g.name)
		if ok == g.unresolved {
			t.Errorf("%s: LookupModel resolved=%v, want %v", g.name, ok, !g.unresolved)
			continue
		}
		if !ok {
			continue // resolution is the fallback's job; covered by TestModelGolden
		}
		t.Run(g.name, func(t *testing.T) {
			if spec.SupportsTemperature != g.temp {
				t.Errorf("SupportsTemperature = %v, want %v", spec.SupportsTemperature, g.temp)
			}
			if spec.DefaultTemperature != g.dtemp {
				t.Errorf("DefaultTemperature = %v, want %v", spec.DefaultTemperature, g.dtemp)
			}
			if spec.MaxOutput != g.maxOut {
				t.Errorf("MaxOutput = %d, want %d", spec.MaxOutput, g.maxOut)
			}
			if spec.UsesMaxCompletionTokens != g.maxComp {
				t.Errorf("UsesMaxCompletionTokens = %v, want %v", spec.UsesMaxCompletionTokens, g.maxComp)
			}
			if spec.SupportsTools != g.tools {
				t.Errorf("SupportsTools = %v, want %v", spec.SupportsTools, g.tools)
			}
			if spec.ContextWindow != g.ctx {
				t.Errorf("ContextWindow = %d, want %d", spec.ContextWindow, g.ctx)
			}
			if spec.CharsPerToken != g.charsTok {
				t.Errorf("CharsPerToken = %v, want %v", spec.CharsPerToken, g.charsTok)
			}
		})
	}
}

// TestGatewayPrefixedModels covers the bug that made "gatorllm/gpt-5.4" (a real
// config in use) fall through to the OpenAI-compatible default and receive
// temperature + max_tokens on a reasoning model.
func TestGatewayPrefixedModels(t *testing.T) {
	t.Run("gatorllm reasoning model keeps reasoning params", func(t *testing.T) {
		d := getModelDefaults("gatorllm/gpt-5.4", "custom")
		if d.SupportsTemperature {
			t.Error("SupportsTemperature = true, want false (reasoning model must not receive temperature)")
		}
		if !d.UsesMaxCompletionTokens {
			t.Error("UsesMaxCompletionTokens = false, want true")
		}
		if d.MaxOutputTokens == 0 {
			t.Error("MaxOutputTokens = 0, want the gpt-5 family cap")
		}
	})

	t.Run("context window matches the bare name", func(t *testing.T) {
		prefixed := getModelContextWindowByName("gatorllm/gpt-5.4")
		bare := getModelContextWindowByName("gpt-5.4")
		if prefixed != bare {
			t.Errorf("prefixed = %d, bare = %d, want equal", prefixed, bare)
		}
	})

	t.Run("openrouter anthropic prefix resolves to the anthropic defaults", func(t *testing.T) {
		d := getModelDefaults("anthropic/claude-sonnet-4-6", "openrouter")
		if d.DefaultTemperature != 1.0 {
			t.Errorf("DefaultTemperature = %v, want 1.0", d.DefaultTemperature)
		}
		if d.MaxOutputTokens != 16384 {
			t.Errorf("MaxOutputTokens = %d, want 16384", d.MaxOutputTokens)
		}
	})

	t.Run("provider overrides still apply after lookup", func(t *testing.T) {
		if d := getModelDefaults("claude-sonnet-4-6", "zai-anthropic"); d.DefaultTemperature != 1.0 {
			t.Errorf("zai-anthropic DefaultTemperature = %v, want 1.0", d.DefaultTemperature)
		}
		if d := getModelDefaults("modelo-custom-local", "ollama"); d.MaxOutputTokens != 4096 {
			t.Errorf("ollama fallback MaxOutputTokens = %d, want 4096", d.MaxOutputTokens)
		}
	})
}

func TestEstimateCostIsDeterministic(t *testing.T) {
	u := NewUsageTracker(nil)
	u.init()
	u.initModelCosts()

	// 1M prompt tokens, no completion: the cost equals InputPer1M.
	t.Run("variant does not inherit the shorter prefix price", func(t *testing.T) {
		got := u.estimateCost("gpt-5-mini", 1_000_000, 0)
		if got != 0.15 {
			t.Errorf("estimateCost(gpt-5-mini) = %v, want 0.15 (gpt-5 would be 2.00)", got)
		}
	})

	t.Run("dated variant falls back to the family price", func(t *testing.T) {
		got := u.estimateCost("gpt-4o-2024-08-06", 1_000_000, 0)
		if got != 2.50 {
			t.Errorf("estimateCost = %v, want 2.50", got)
		}
	})

	t.Run("gateway prefix is priced", func(t *testing.T) {
		if got := u.estimateCost("gatorllm/gpt-5", 1_000_000, 0); got != 2.00 {
			t.Errorf("estimateCost = %v, want 2.00", got)
		}
	})

	t.Run("user configured cost wins over the registry", func(t *testing.T) {
		u2 := NewUsageTracker(nil)
		u2.init()
		u2.modelCosts["gpt-5"] = ModelCost{InputPer1M: 99.0, OutputPer1M: 99.0}
		u2.initModelCosts()
		if got := u2.estimateCost("gpt-5", 1_000_000, 0); got != 99.0 {
			t.Errorf("estimateCost = %v, want 99.0 (user config must win)", got)
		}
	})

	t.Run("unknown model costs nothing", func(t *testing.T) {
		if got := u.estimateCost("modelo-inexistente", 1_000_000, 0); got != 0 {
			t.Errorf("estimateCost = %v, want 0", got)
		}
	})
}

func TestCharsPerTokenPrefixed(t *testing.T) {
	if got, want := charsPerToken("gatorllm/gpt-5.4"), charsPerToken("gpt-5.4"); got != want {
		t.Errorf("charsPerToken(gatorllm/gpt-5.4) = %v, want %v", got, want)
	}
	// Unknown names must still reach the family heuristic.
	if got := charsPerToken("algum-claude-customizado"); got != 3.5 {
		t.Errorf("charsPerToken(custom claude) = %v, want 3.5 from the fallback", got)
	}
}

func TestNewOpenAIModels(t *testing.T) {
	cases := []struct {
		name   string
		ctx    int
		effort []string
	}{
		{"gpt-6-astra", 1050000, effortGPT6},
		{"gpt-5.6", 1050000, effortGPT5},
		{"gpt-5.6-sol", 1050000, effortGPT5},
		{"gpt-5.6-terra", 1050000, effortGPT5},
		{"gpt-5.6-luna", 1050000, effortGPT5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := getModelContextWindowByName(c.name); got != c.ctx {
				t.Errorf("contextWindow = %d, want %d", got, c.ctx)
			}
			d := getModelDefaults(c.name, "openai")
			if d.SupportsTemperature {
				t.Error("SupportsTemperature = true, want false (reasoning model)")
			}
			if !d.UsesMaxCompletionTokens {
				t.Error("UsesMaxCompletionTokens = false, want true")
			}
			if d.MaxOutputTokens != 16384 {
				t.Errorf("MaxOutputTokens = %d, want 16384", d.MaxOutputTokens)
			}
			spec, ok := LookupModel(c.name)
			if !ok {
				t.Fatal("not found in registry")
			}
			if len(spec.EffortLevels) != len(c.effort) {
				t.Errorf("EffortLevels = %v, want %v", spec.EffortLevels, c.effort)
			}
		})
	}

	t.Run("gpt-6-astra pricing", func(t *testing.T) {
		spec, _ := LookupModel("gpt-6-astra")
		if spec.InputPer1M != 10.00 || spec.OutputPer1M != 50.00 {
			t.Errorf("pricing = %v/%v, want 10.00/50.00", spec.InputPer1M, spec.OutputPer1M)
		}
	})

	t.Run("gpt-5.6 has no unverified price", func(t *testing.T) {
		spec, _ := LookupModel("gpt-5.6")
		if spec.InputPer1M != 0 || spec.OutputPer1M != 0 {
			t.Errorf("pricing = %v/%v, want 0/0 (not published)", spec.InputPer1M, spec.OutputPer1M)
		}
	})

	t.Run("older gpt-5 variants do not regress", func(t *testing.T) {
		for _, n := range []string{"gpt-5.4", "gpt-5.5", "gpt-5.2"} {
			if got := getModelContextWindowByName(n); got != 128000 {
				t.Errorf("%s contextWindow = %d, want 128000", n, got)
			}
		}
	})
}

func TestClaude5Family(t *testing.T) {
	// Claude 5 ships the 1M window ungated; the 4.x family needs the
	// context-1m beta header, so it stays at the safe 200k floor.
	noSampling := []string{
		"claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-fable-5-1",
	}
	noSamplingGated := []string{"claude-opus-4-7", "claude-opus-4-8"}
	for _, n := range noSampling {
		t.Run(n+" rejects temperature", func(t *testing.T) {
			if d := getModelDefaults(n, "anthropic"); d.SupportsTemperature {
				t.Error("SupportsTemperature = true, want false (400 on Claude 4.7+)")
			}
			if got := getModelContextWindowByName(n); got != 1000000 {
				t.Errorf("contextWindow = %d, want 1000000", got)
			}
		})
	}

	for _, n := range noSamplingGated {
		t.Run(n+" rejects temperature at the gated window", func(t *testing.T) {
			if d := getModelDefaults(n, "anthropic"); d.SupportsTemperature {
				t.Error("SupportsTemperature = true, want false (400 on Claude 4.7+)")
			}
			if got := getModelContextWindowByName(n); got != 200000 {
				t.Errorf("contextWindow = %d, want 200000 (1M needs the context-1m beta)", got)
			}
		})
	}

	t.Run("older models keep temperature", func(t *testing.T) {
		for _, n := range []string{"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5", "claude-3-opus"} {
			if d := getModelDefaults(n, "anthropic"); !d.SupportsTemperature {
				t.Errorf("%s SupportsTemperature = false, want true", n)
			}
		}
	})

	t.Run("haiku 4.5 context", func(t *testing.T) {
		if got := getModelContextWindowByName("claude-haiku-4-5"); got != 200000 {
			t.Errorf("contextWindow = %d, want 200000", got)
		}
	})

	t.Run("gateway prefixed opus-5", func(t *testing.T) {
		if d := getModelDefaults("anthropic/claude-opus-5", "openrouter"); d.SupportsTemperature {
			t.Error("SupportsTemperature = true, want false")
		}
	})

	t.Run("tokenizer ratio drops from 4.7 on", func(t *testing.T) {
		if got := charsPerToken("claude-opus-5"); got != 3.0 {
			t.Errorf("charsPerToken(claude-opus-5) = %v, want 3.0", got)
		}
		if got := charsPerToken("claude-sonnet-4-6"); got != 3.5 {
			t.Errorf("charsPerToken(claude-sonnet-4-6) = %v, want 3.5", got)
		}
	})

	t.Run("pricing", func(t *testing.T) {
		want := map[string][2]float64{
			"claude-fable-5-1":  {10.00, 50.00},
			"claude-opus-5":     {5.00, 25.00},
			"claude-opus-4-8":   {5.00, 25.00},
			"claude-opus-4-7":   {5.00, 25.00},
			"claude-sonnet-5":   {2.00, 10.00},
			"claude-sonnet-4-6": {3.00, 15.00},
			"claude-haiku-4-5":  {1.00, 5.00},
		}
		for n, w := range want {
			spec, ok := LookupModel(n)
			if !ok {
				t.Fatalf("%s not in registry", n)
			}
			if spec.InputPer1M != w[0] || spec.OutputPer1M != w[1] {
				t.Errorf("%s pricing = %v/%v, want %v/%v", n, spec.InputPer1M, spec.OutputPer1M, w[0], w[1])
			}
		}
	})
}

// TestAnthropicPayloadOmitsTemperature exercises the real request path: with
// Claude 4.7+ the temperature key must not reach the wire at all.
func TestAnthropicPayloadOmitsTemperature(t *testing.T) {
	build := func(model string) string {
		temp, maxTok := anthropicSamplingFor(model, "anthropic")
		req := convertToAnthropicRequest(model, []chatMessage{{Role: "user", Content: "oi"}}, nil, temp, &maxTok)
		b, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return string(b)
	}

	if body := build("claude-opus-5"); strings.Contains(body, "temperature") {
		t.Errorf("claude-opus-5 payload carries temperature: %s", body)
	}
	if body := build("claude-sonnet-4-6"); !strings.Contains(body, "temperature") {
		t.Errorf("claude-sonnet-4-6 payload lost temperature: %s", body)
	}
}

func TestReasoningEffort(t *testing.T) {
	c := &LLMClient{provider: "openai"}

	t.Run("supported level passes", func(t *testing.T) {
		if got := c.reasoningEffortFor("gpt-6-astra", "xhigh"); got != "xhigh" {
			t.Errorf("got %q, want xhigh", got)
		}
		if got := c.reasoningEffortFor("gpt-5.6-sol", "max"); got != "max" {
			t.Errorf("got %q, want max", got)
		}
	})

	t.Run("unsupported level is dropped", func(t *testing.T) {
		// gpt-6-astra does not offer "none" or "low".
		if got := c.reasoningEffortFor("gpt-6-astra", "low"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
		if got := c.reasoningEffortFor("gpt-5.6", "turbo"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("model that does not take the param drops it", func(t *testing.T) {
		if got := c.reasoningEffortFor("gpt-4o", "low"); got != "" {
			t.Errorf("got %q, want empty (gpt-4o rejects reasoning_effort)", got)
		}
	})

	t.Run("reasoning model without declared levels passes through", func(t *testing.T) {
		// gpt-5 and the o-series take the param; the exact level set is not pinned.
		if got := c.reasoningEffortFor("gpt-5", "high"); got != "high" {
			t.Errorf("got %q, want high", got)
		}
		if got := c.reasoningEffortFor("o3-mini", "low"); got != "low" {
			t.Errorf("got %q, want low", got)
		}
	})

	t.Run("unknown model passes through", func(t *testing.T) {
		if got := c.reasoningEffortFor("", "low"); got != "low" {
			t.Errorf("got %q, want low", got)
		}
		if got := c.reasoningEffortFor("modelo-custom", "low"); got != "low" {
			t.Errorf("got %q, want low", got)
		}
	})

	t.Run("configured effort wins over fast mode", func(t *testing.T) {
		client := &LLMClient{provider: "openai", params: map[string]any{"reasoning_effort": "max"}}
		req := &chatRequest{Model: "gpt-6-astra"}
		client.applyModelDefaults(req)
		client.applyFastMode(ContextWithFastMode(context.Background(), true), req)
		if req.ReasoningEffort != "max" {
			t.Errorf("ReasoningEffort = %q, want max", req.ReasoningEffort)
		}
		if req.ServiceTier != "priority" {
			t.Errorf("ServiceTier = %q, want priority", req.ServiceTier)
		}
	})

	t.Run("fast mode falls back to the cheapest offered level", func(t *testing.T) {
		client := &LLMClient{provider: "openai"}
		req := &chatRequest{Model: "gpt-6-astra"}
		client.applyFastMode(ContextWithFastMode(context.Background(), true), req)
		// gpt-6-astra has no "low"; dropping the field left the user on premium
		// pricing with no latency gain, so it takes the lowest declared level.
		if req.ReasoningEffort != "medium" {
			t.Errorf("ReasoningEffort = %q, want medium", req.ReasoningEffort)
		}
	})

	t.Run("fast mode prefers low when offered", func(t *testing.T) {
		client := &LLMClient{provider: "openai"}
		req := &chatRequest{Model: "gpt-5.6"}
		client.applyFastMode(ContextWithFastMode(context.Background(), true), req)
		if req.ReasoningEffort != "low" {
			t.Errorf("ReasoningEffort = %q, want low", req.ReasoningEffort)
		}
	})

	t.Run("fast mode omits the field on a model that rejects it", func(t *testing.T) {
		client := &LLMClient{provider: "openai"}
		req := &chatRequest{Model: "gpt-4o"}
		client.applyFastMode(ContextWithFastMode(context.Background(), true), req)
		if req.ReasoningEffort != "" {
			t.Errorf("ReasoningEffort = %q, want empty", req.ReasoningEffort)
		}
		if req.ServiceTier != "priority" {
			t.Errorf("ServiceTier = %q, want priority", req.ServiceTier)
		}
	})
}

// TestPriceFallback covers the regression where a newer entry without a
// published price shadowed the priced family prefix and reported zero cost.
func TestPriceFallback(t *testing.T) {
	u := NewUsageTracker(nil)
	u.init()
	u.initModelCosts()

	cases := map[string]float64{
		"gpt-5.6":     2.00, // own entry has no price -> gpt-5 family
		"gpt-5.6-sol": 2.00,
		"glm-5.1":     1.00, // own entry has no price -> glm-5 family
		"gpt-5.4":     2.00,
		"gpt-6-astra": 10.00, // has its own price
	}
	for model, want := range cases {
		if got := u.estimateCost(model, 1_000_000, 0); got != want {
			t.Errorf("estimateCost(%s) = %v, want %v", model, got, want)
		}
	}

	if _, _, ok := LookupModelPrice("modelo-inexistente"); ok {
		t.Error("LookupModelPrice resolved an unknown model")
	}
}

// TestAnthropic1MBetaGate pins the beta-header check to the registry so the
// header and the reported context window cannot drift apart.
func TestAnthropic1MBetaGate(t *testing.T) {
	cases := map[string]bool{
		"claude-opus-4-8":           true,
		"claude-opus-4-7":           true,
		"claude-sonnet-4-6":         true,
		"claude-opus-4-20250514":    true,
		"anthropic/claude-opus-4-8": true,  // gateway prefix: the old HasPrefix missed this
		"claude-opus-5":             false, // 1M is native, no beta header
		"claude-sonnet-5":           false,
		"claude-3-opus":             false,
		"gpt-5":                     false,
	}
	for model, want := range cases {
		if got := isAnthropic1MModel(model); got != want {
			t.Errorf("isAnthropic1MModel(%s) = %v, want %v", model, got, want)
		}
	}

	// A model that advertises a beta window must not report it as the default.
	spec, _ := LookupModel("claude-opus-4-8")
	if spec.ContextWindow != 200000 || spec.BetaContextWindow != 1000000 {
		t.Errorf("claude-opus-4-8 = %d/%d, want 200000/1000000", spec.ContextWindow, spec.BetaContextWindow)
	}
}

// TestSegmentBoundary is the unit-level guard for the folding collision.
func TestSegmentBoundary(t *testing.T) {
	if prefixMatches("gpt-4-1106-preview", "gpt-4-1") {
		t.Error("gpt-4-1 must not match gpt-4-1106-preview")
	}
	for _, c := range []struct{ name, key string }{
		{"gpt-4-1106-preview", "gpt-4"},
		{"gpt-5-mini", "gpt-5"},
		{"claude-3-5-sonnet", "claude-3"},
		{"llama3", "llama"},
		{"o1-preview", "o1"},
		{"glm-5-1", "glm-5"},
	} {
		if !prefixMatches(c.name, c.key) {
			t.Errorf("prefixMatches(%q, %q) = false, want true", c.name, c.key)
		}
	}
}

func TestGLMNewModels(t *testing.T) {
	cases := map[string][2]float64{
		"glm-5.2":       {1.40, 4.40},
		"glm-5.3":       {1.40, 4.40},
		"glm-5.3-flash": {0.15, 0.50},
	}
	for n, price := range cases {
		t.Run(n, func(t *testing.T) {
			if got := getModelContextWindowByName(n); got != 1048576 {
				t.Errorf("contextWindow = %d, want 1048576", got)
			}
			spec, ok := LookupModel(n)
			if !ok || spec.Canonical != n {
				t.Fatalf("resolved to %q (ok=%v), want %s", spec.Canonical, ok, n)
			}
			if spec.InputPer1M != price[0] || spec.OutputPer1M != price[1] {
				t.Errorf("pricing = %v/%v, want %v/%v", spec.InputPer1M, spec.OutputPer1M, price[0], price[1])
			}
			if d := getModelDefaults(n, "zai"); !d.SupportsTemperature {
				t.Error("SupportsTemperature = false, want true")
			}
		})
	}

	t.Run("flash is not shadowed by glm-5.3", func(t *testing.T) {
		spec, _ := LookupModel("glm-5.3-flash")
		if spec.InputPer1M != 0.15 {
			t.Errorf("InputPer1M = %v, want 0.15 (must not inherit glm-5.3 pricing)", spec.InputPer1M)
		}
	})

	t.Run("reasoning effort levels", func(t *testing.T) {
		c := &LLMClient{provider: "zai"}
		if got := c.reasoningEffortFor("glm-5.3", "max"); got != "max" {
			t.Errorf("got %q, want max", got)
		}
		// GLM offers low/high/max — "medium" is not one of them.
		if got := c.reasoningEffortFor("glm-5.3", "medium"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}
