// Package copilot – usage_tracker.go records LLM token usage and estimated costs
// per session and globally.
package copilot

import (
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// ModelCost holds pricing per 1M tokens for a model.
type ModelCost struct {
	InputPer1M  float64 `yaml:"input_per_1m"`  // USD per 1M input tokens
	OutputPer1M float64 `yaml:"output_per_1m"` // USD per 1M output tokens
}

// SessionUsage holds token and cost stats for a session.
type SessionUsage struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
	Requests         int64
	EstimatedCostUSD float64
	FirstRequestAt   time.Time
	LastRequestAt    time.Time
}

// UsageTracker records usage per session and globally.
type UsageTracker struct {
	mu sync.RWMutex

	sessions   map[string]*SessionUsage
	global     *SessionUsage
	modelCosts map[string]ModelCost
	costsOnce  sync.Once

	logger *slog.Logger
}

// NewUsageTracker creates a new UsageTracker.
func NewUsageTracker(logger *slog.Logger) *UsageTracker {
	if logger == nil {
		logger = slog.Default()
	}
	return &UsageTracker{
		sessions:   make(map[string]*SessionUsage),
		global:     &SessionUsage{},
		modelCosts: make(map[string]ModelCost),
		logger:     logger.With("component", "usage_tracker"),
	}
}

func (u *UsageTracker) init() {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.modelCosts == nil {
		u.modelCosts = make(map[string]ModelCost)
	}
	if u.sessions == nil {
		u.sessions = make(map[string]*SessionUsage)
	}
	if u.global == nil {
		u.global = &SessionUsage{}
	}
}

// initModelCosts seeds prices from the model registry, leaving any cost the
// user configured untouched.
func (u *UsageTracker) initModelCosts() {
	u.costsOnce.Do(func() {
		for _, spec := range modelRegistry {
			if spec.InputPer1M == 0 && spec.OutputPer1M == 0 {
				continue
			}
			if _, ok := u.modelCosts[spec.Canonical]; !ok {
				u.modelCosts[spec.Canonical] = ModelCost{InputPer1M: spec.InputPer1M, OutputPer1M: spec.OutputPer1M}
			}
		}
	})
}

// Record adds usage for a session and globally.
func (u *UsageTracker) Record(sessionID, model string, usage LLMUsage) {
	u.init()
	u.mu.Lock()
	defer u.mu.Unlock()
	u.initModelCosts()

	now := time.Now()

	// Session
	su, ok := u.sessions[sessionID]
	if !ok {
		su = &SessionUsage{FirstRequestAt: now}
		u.sessions[sessionID] = su
	}
	su.PromptTokens += int64(usage.PromptTokens)
	su.CompletionTokens += int64(usage.CompletionTokens)
	su.TotalTokens += int64(usage.TotalTokens)
	su.Requests++
	su.LastRequestAt = now

	cost := u.estimateCost(model, usage.PromptTokens, usage.CompletionTokens)
	su.EstimatedCostUSD += cost

	// Global
	u.global.PromptTokens += int64(usage.PromptTokens)
	u.global.CompletionTokens += int64(usage.CompletionTokens)
	u.global.TotalTokens += int64(usage.TotalTokens)
	u.global.Requests++
	if u.global.FirstRequestAt.IsZero() {
		u.global.FirstRequestAt = now
	}
	u.global.LastRequestAt = now
	u.global.EstimatedCostUSD += cost
}

func (u *UsageTracker) estimateCost(model string, prompt, completion int) float64 {
	cost, ok := u.modelCosts[model]
	if !ok {
		// Registry resolution handles variants, gateway prefixes and dotted
		// spellings with a deterministic longest-prefix match. The old loop
		// walked the map, so "gpt-5-mini" could pick up "gpt-5" pricing
		// depending on iteration order.
		if in, out, found := LookupModelPrice(model); found {
			cost = ModelCost{InputPer1M: in, OutputPer1M: out}
			ok = true
		}
	}
	if !ok {
		return 0
	}
	return (float64(prompt)/1e6)*cost.InputPer1M + (float64(completion)/1e6)*cost.OutputPer1M
}

// GetSession returns a copy of the session's usage stats, or nil if not found.
func (u *UsageTracker) GetSession(sessionID string) *SessionUsage {
	u.mu.RLock()
	defer u.mu.RUnlock()

	su, ok := u.sessions[sessionID]
	if !ok {
		return nil
	}
	return &SessionUsage{
		PromptTokens:     su.PromptTokens,
		CompletionTokens: su.CompletionTokens,
		TotalTokens:      su.TotalTokens,
		Requests:         su.Requests,
		EstimatedCostUSD: su.EstimatedCostUSD,
		FirstRequestAt:   su.FirstRequestAt,
		LastRequestAt:    su.LastRequestAt,
	}
}

// GetGlobal returns a copy of global usage.
func (u *UsageTracker) GetGlobal() *SessionUsage {
	u.mu.RLock()
	defer u.mu.RUnlock()

	g := u.global
	return &SessionUsage{
		PromptTokens:     g.PromptTokens,
		CompletionTokens: g.CompletionTokens,
		TotalTokens:      g.TotalTokens,
		Requests:         g.Requests,
		EstimatedCostUSD: g.EstimatedCostUSD,
		FirstRequestAt:   g.FirstRequestAt,
		LastRequestAt:    g.LastRequestAt,
	}
}

// ResetSession clears usage for a session.
func (u *UsageTracker) ResetSession(sessionID string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	delete(u.sessions, sessionID)
}

// FormatUsage returns a human-readable usage report for a session.
func (u *UsageTracker) FormatUsage(sessionID string) string {
	su := u.GetSession(sessionID)
	if su == nil {
		return fmt.Sprintf("No usage recorded for session %s.", sessionID)
	}
	return formatSessionUsage(sessionID, su)
}

// FormatGlobalUsage returns a human-readable global usage report.
func (u *UsageTracker) FormatGlobalUsage() string {
	g := u.GetGlobal()
	return formatSessionUsage("global", g)
}

func formatSessionUsage(label string, su *SessionUsage) string {
	var b string
	if su.Requests == 0 {
		b = fmt.Sprintf("*Usage (%s)*\n\nNo requests yet.", label)
		return b
	}
	b = fmt.Sprintf("*Usage (%s)*\n\n", label)
	b += fmt.Sprintf("Prompt tokens: %d\n", su.PromptTokens)
	b += fmt.Sprintf("Completion tokens: %d\n", su.CompletionTokens)
	b += fmt.Sprintf("Total tokens: %d\n", su.TotalTokens)
	b += fmt.Sprintf("Requests: %d\n", su.Requests)
	b += fmt.Sprintf("Est. cost: $%.4f\n", su.EstimatedCostUSD)
	if !su.FirstRequestAt.IsZero() {
		b += fmt.Sprintf("First request: %s\n", su.FirstRequestAt.Format("2006-01-02 15:04"))
	}
	if !su.LastRequestAt.IsZero() {
		b += fmt.Sprintf("Last request: %s", su.LastRequestAt.Format("2006-01-02 15:04"))
	}
	return b
}
