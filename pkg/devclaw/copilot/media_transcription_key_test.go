package copilot

import "testing"

// TestTranscriptionKeyResolvesFromEnv covers the WhatsApp audio bug: the vault
// exports every entry as an env var, but nothing carried one onto the media
// config, so a key stored the way CLAUDE.md requires never reached Whisper.
func TestTranscriptionKeyResolvesFromEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test-from-vault")

	cfg := &Config{}
	ResolveSecrets(cfg)

	if cfg.Media.TranscriptionAPIKey != "sk-test-from-vault" {
		t.Errorf("TranscriptionAPIKey = %q, want the env value", cfg.Media.TranscriptionAPIKey)
	}
}

func TestTranscriptionKeyKeepsExplicitValue(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-from-env")

	cfg := &Config{}
	cfg.Media.TranscriptionAPIKey = "sk-explicit"
	ResolveSecrets(cfg)

	if cfg.Media.TranscriptionAPIKey != "sk-explicit" {
		t.Errorf("TranscriptionAPIKey = %q, want the explicit value to win", cfg.Media.TranscriptionAPIKey)
	}
}

// TestResolveForProviderRespectsConfiguredKey pins the other half of the bug:
// deriving an endpoint from the main provider while a dedicated key is set
// ships that key to a host it does not belong to.
func TestResolveForProviderRespectsConfiguredKey(t *testing.T) {
	m := MediaConfig{TranscriptionAPIKey: "sk-openai"}
	m.ResolveForProvider("zai", "https://api.z.ai/api/paas/v4")

	if m.TranscriptionBaseURL != "" {
		t.Errorf("TranscriptionBaseURL = %q, want empty: an OpenAI key must not be pointed at Z.AI",
			m.TranscriptionBaseURL)
	}

	// With no key configured the auto-derivation must still work as before.
	m2 := MediaConfig{}
	m2.ResolveForProvider("zai", "https://api.z.ai/api/paas/v4")
	if m2.TranscriptionBaseURL == "" {
		t.Error("TranscriptionBaseURL = empty, want the derived Z.AI endpoint (no regression)")
	}
}
