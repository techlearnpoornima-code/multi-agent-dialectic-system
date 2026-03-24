# MADS — Multi-Agent Dialectic System v1.0

> A LangGraph + Ollama pipeline that forces two LLMs into a Steel-Man
> debate loop and synthesises the result using Hegelian Dialectics.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        LANGGRAPH STATE           │
                    │  thread_id, topic, transcript,   │
                    │  fact_check_logs, synthesis      │
                    └─────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     AG_001 ORCHESTRATOR      │
                    │  Generates extreme personas  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │                                            │
    ┌─────────▼──────────┐                   ┌────────────▼────────┐
    │  AG_002 PRO-AGENT  │                   │  AG_003 CON-AGENT   │
    │  Model: llama3.2   │                   │  Model: mistral     │
    │  Thesis / Opening  │◄─────────────────►│  Antithesis         │
    └─────────┬──────────┘   Steel-Man x2    └────────────┬────────┘
              │                                            │
              └────────────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   AG_004 FACT-CHECKER        │
                    │  Passive fallacy monitor     │
                    │  Scans every 2 turns         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   AG_005 SYNTHESIZER         │
                    │  Hegelian Thesis+Antithesis  │
                    │  → Synthesis framework       │
                    └─────────────────────────────┘
```

## Execution Flow

```
orchestrate → pro_opening → con_opening → fact_check(r1)
           → pro_rebuttal(r1) → con_rebuttal(r1) → fact_check(r2)
           → pro_rebuttal(r2) → con_rebuttal(r2)
           → synthesize → done
```

---

## Setup

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

### 2. Pull Models

```bash
# Minimum: one model works for all roles
ollama pull llama3.2

# Recommended: dedicated models per role (better friction)
ollama pull llama3.2      # Orchestrator, Pro-Agent, Fact-Checker, Synthesizer
ollama pull mistral       # Con-Agent (different reasoning style)

# Alternatives
ollama pull qwen2.5       # Strong for Con role
ollama pull gemma3        # Good for Synthesizer
ollama pull phi4          # Lightweight option
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Interactive CLI
```bash
python run.py
```

### Direct with topic
```bash
python run.py "Nuclear energy is essential for climate change mitigation"
python run.py "Social media causes more harm than good to democracy"
python run.py "Consciousness is purely a product of physical brain processes"
```

### Programmatic
```python
from mads import run_debate

state = run_debate("Universal Basic Income should replace all welfare programs")

print(state["final_synthesis"])
print(state["transcript"])
print(state["fact_check_logs"])
```

---

## Configuration

Edit the `MODELS` dict in `mads.py` to use different LLMs per role:

```python
MODELS = {
    "orchestrator": "llama3.2",    # Generates personas & setup
    "pro":          "llama3.2",    # Thesis / Pro-Agent
    "con":          "mistral",     # Antithesis / Con-Agent  ← change this
    "fact_checker": "llama3.2",    # Logic & fallacy monitor
    "synthesizer":  "llama3.2",    # Final Hegelian synthesis
}
```

**Pro tip**: Use models with distinct "personalities" for Pro vs Con:
- `llama3.2` vs `mistral` — different base training
- `qwen2.5` vs `gemma3` — different reasoning styles
- Any local quantized GGUF via Ollama

---

## Output Files

After each run, two files are saved:

| File | Contents |
|------|----------|
| `mads_report_<id>.txt` | Full formatted debate transcript + synthesis |
| `mads_state_<id>.json` | Raw JSON state (transcript, logs, personas, synthesis) |

---

## State Schema

```json
{
  "thread_id": "uuid-v4",
  "topic": "string",
  "pro_persona": "Expert bio paragraph",
  "con_persona": "Expert bio paragraph",
  "transcript": [
    {
      "turn": 1,
      "agent": "Pro",
      "phase": "opening",
      "content": "...",
      "timestamp": "ISO-8601"
    }
  ],
  "fact_check_logs": [
    {
      "turn_ref": 2,
      "round": 1,
      "fallacies_found": [],
      "suspect_claims": [],
      "verdict": "CLEAN | MINOR_ISSUES | MAJOR_ISSUES"
    }
  ],
  "steel_man_round": 2,
  "phase": "done",
  "final_synthesis": "Markdown synthesis text"
}
```

---

## Steel-Man Protocol

Each rebuttal turn follows a mandatory 3-step structure:

```
## 1. VALIDATION
   State opponent's strongest argument accurately.

## 2. AMPLIFICATION
   Add logic/data that makes their argument even stronger.
   (Forces intellectual honesty — no straw-manning allowed)

## 3. REFUTATION
   Explain why your framework remains superior despite that truth.
```

---

## The LangGraph State Machine

```
Nodes:  orchestrator → pro_opening → con_opening → fact_checker
        → pro_rebuttal → con_rebuttal → (loop×2) → synthesizer → END

Edges:  All conditional, routed by state["phase"] field
        Steel-Man loop: steel_man_round counter 0→1→2
```

---

## Extending MADS

### Add a 3rd debater (Moderator)
```python
builder.add_node("moderator", moderator_node)
# Insert after each rebuttal pair
```

### Stream responses in real-time
```python
for chunk in graph.stream(initial_state):
    print(chunk)
```

### Custom fact-check rules
Edit the `fact_checker_node` prompt to add domain-specific checks.

### Export to PDF / HTML
Use the `final_synthesis` field from the returned state.
