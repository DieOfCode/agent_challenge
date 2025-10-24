package agent

// High-level pair agents description:
// AgentOne produces a JSON payload; AgentTwo consumes that JSON and produces a final text.

type AgentOneTask struct {
    Goal       string `json:"goal"`
    Input      string `json:"input"`
    OutputJSON bool   `json:"output_json"`
}

type AgentOneResult struct {
    Summary   string   `json:"summary"`
    Findings  []string `json:"findings"`
    NextSteps []string `json:"next_steps"`
}

type AgentTwoTask struct {
    Instruction string        `json:"instruction"`
    Data        AgentOneResult `json:"data"`
}


