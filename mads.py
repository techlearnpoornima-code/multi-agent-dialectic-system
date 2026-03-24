"""
Multi-Agent Dialectic System (MADS) v1.0
=========================================
Hegelian Thesis → Antithesis → Synthesis debate engine
built on LangGraph + Ollama with dedicated LLMs per role.

Architecture:
  Orchestrator  ──►  Pro-Agent (LLM-A)  ──►  Con-Agent (LLM-B)
       │                    │                       │
       │            Steel-Man Loop (2×)             │
       │                    └───────────────────────┘
       │
  Fact-Checker (passive observer)
       │
  Synthesizer  ──►  Final Markdown Report
"""

from __future__ import annotations

import json
import uuid
import textwrap
from datetime import datetime
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ─────────────────────────────────────────────
#  MODEL CONFIGURATION
#  Edit these to match the models you have pulled
# ─────────────────────────────────────────────
MODELS = {
    "orchestrator": "llama3.2:latest",   # Generates personas & setup
    "pro":          "llama3.2:latest",   # Thesis / Pro-Agent  (swap to a different model if available)
    "con":          "mistral:latest",    # Antithesis / Con-Agent
    "fact_checker": "llama3.2:latest",   # Passive logic monitor
    "synthesizer":  "llama3.2:latest",   # Final Hegelian synthesis
}

OLLAMA_BASE = "http://localhost:11434"


# ─────────────────────────────────────────────
#  STATE SCHEMA
# ─────────────────────────────────────────────
class DebateState(TypedDict):
    thread_id: str
    topic: str
    pro_persona: str
    con_persona: str
    transcript: list[dict]          # {turn, agent, phase, content}
    fact_check_logs: list[dict]     # {turn_ref, flags}
    steel_man_round: int            # 0 → 1 → 2
    phase: Literal[
        "orchestrate",
        "pro_opening", "con_opening",
        "pro_rebuttal", "con_rebuttal",
        "fact_check",
        "synthesize",
        "done"
    ]
    final_synthesis: str


# ─────────────────────────────────────────────
#  LLM FACTORY
# ─────────────────────────────────────────────
def get_llm(role: str, temperature: float = 0.7) -> ChatOllama:
    model = MODELS[role]
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE,
        temperature=temperature,
    )


# ─────────────────────────────────────────────
#  HELPER: pretty transcript entry
# ─────────────────────────────────────────────
def add_turn(state: DebateState, agent: str, phase: str, content: str) -> list[dict]:
    turn_num = len(state["transcript"]) + 1
    entry = {
        "turn": turn_num,
        "agent": agent,
        "phase": phase,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    return state["transcript"] + [entry]


def last_two_turns(state: DebateState) -> str:
    recent = state["transcript"][-2:]
    return "\n\n".join(
        f"[{t['agent'].upper()} – {t['phase']}]\n{t['content']}"
        for t in recent
    )


def full_transcript_text(state: DebateState) -> str:
    return "\n\n".join(
        f"--- Turn {t['turn']} | {t['agent'].upper()} | {t['phase']} ---\n{t['content']}"
        for t in state["transcript"]
    )


# ─────────────────────────────────────────────
#  NODE 1 – ORCHESTRATOR
#  Generates extreme expert personas
# ─────────────────────────────────────────────
def orchestrator_node(state: DebateState) -> DebateState:
    print("\n[ORCHESTRATOR] Generating expert personas...")
    llm = get_llm("orchestrator", temperature=0.9)

    prompt = f"""You are a debate architect. For the topic statement: "{state['topic']}"

The PRO side believes this statement is TRUE and will argue FOR it.
The CON side believes this statement is FALSE and will argue AGAINST it.

Generate TWO extreme, hyper-specialized expert personas:
- PRO expert: fiercely believes the topic statement IS TRUE and advocates for it
- CON expert: fiercely believes the topic statement IS FALSE and argues against it

Return ONLY valid JSON with this exact structure:
{{
  "pro": "One paragraph describing the Pro expert's name, background, and why they passionately believe the topic statement is TRUE",
  "con": "One paragraph describing the Con expert's name, background, and why they passionately believe the topic statement is FALSE"
}}

Make the personas vivid, credentialed, and ideologically extreme but intellectually serious.
Ensure the two personas genuinely OPPOSE each other on this specific topic."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Extract JSON robustly
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        personas = json.loads(raw[start:end])
        pro_persona = personas["pro"]
        con_persona = personas["con"]
    except Exception:
        pro_persona = f"Dr. Marcus Vale, a fervent advocate for {state['topic']} with 20 years of research."
        con_persona = f"Prof. Lyra Chen, a seasoned critic of {state['topic']} known for rigorous counter-arguments."

    print(f"  PRO → {pro_persona[:80]}...")
    print(f"  CON → {con_persona[:80]}...")

    return {
        **state,
        "pro_persona": pro_persona,
        "con_persona": con_persona,
        "phase": "pro_opening",
    }


# ─────────────────────────────────────────────
#  NODE 2 – PRO OPENING (Thesis)
# ─────────────────────────────────────────────
def pro_opening_node(state: DebateState) -> DebateState:
    print("\n[PRO-AGENT] Delivering opening thesis...")
    llm = get_llm("pro", temperature=0.75)

    system = f"""You ARE this expert — speak in first person:
{state['pro_persona']}

You are arguing STRONGLY IN FAVOR of: "{state['topic']}"
Be aggressive, use domain-specific evidence, cite mechanisms and data.
Write approximately 300 words. No hedging. This is your thesis."""

    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content=f"Deliver your opening argument FOR: {state['topic']}")])

    content = response.content.strip()
    transcript = add_turn(state, "Pro", "opening", content)

    print(f"  → {content[:120]}...")
    return {**state, "transcript": transcript, "phase": "con_opening"}


# ─────────────────────────────────────────────
#  NODE 3 – CON OPENING (Antithesis)
# ─────────────────────────────────────────────
def con_opening_node(state: DebateState) -> DebateState:
    print("\n[CON-AGENT] Delivering opening antithesis...")
    llm = get_llm("con", temperature=0.75)

    pro_opening = state["transcript"][-1]["content"]

    system = f"""You ARE this expert — speak in first person:
{state['con_persona']}

CRITICAL INSTRUCTION — YOUR ASSIGNED POSITION:
The debate topic is: "{state['topic']}"
The PRO side AGREES with this statement and argues it is TRUE.
YOU are the CON side. You DISAGREE with this statement. You must argue it is FALSE or WRONG.
Your job is to OPPOSE and REFUTE the pro position with aggressive counter-arguments.

Do NOT agree with the pro side under any circumstances.
Do NOT argue the same points as the pro side.
Be aggressive, use domain-specific counter-evidence and systemic critiques.
Write approximately 300 words. No hedging. This is your antithesis."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""The PRO side has just argued IN FAVOUR of: "{state['topic']}"
Their argument:

{pro_opening}

Now YOU must deliver YOUR opening argument OPPOSING this position.
You believe the statement "{state['topic']}" is FALSE or HARMFUL.
Argue forcefully AGAINST it using evidence, logic, and expert reasoning.""")
    ])

    content = response.content.strip()
    transcript = add_turn(state, "Con", "opening", content)

    print(f"  → {content[:120]}...")
    return {**state, "transcript": transcript, "phase": "fact_check", "steel_man_round": 1}


# ─────────────────────────────────────────────
#  NODE 4 – FACT CHECKER (Passive Observer)
# ─────────────────────────────────────────────
def fact_checker_node(state: DebateState) -> DebateState:
    print(f"\n[FACT-CHECKER] Scanning last 2 turns for fallacies...")
    llm = get_llm("fact_checker", temperature=0.2)

    recent = last_two_turns(state)
    turn_ref = len(state["transcript"])

    prompt = f"""You are a logic and fact auditor. Scan the following debate turns for:
1. Named logical fallacies (ad hominem, straw man, false dichotomy, etc.)
2. Unverifiable statistical claims or hallucinated citations
3. Internal contradictions

Debate turns:
{recent}

Return a concise JSON object:
{{
  "fallacies_found": ["list of fallacies with the offending sentence"],
  "suspect_claims": ["list of claims that seem fabricated or unverifiable"],
  "verdict": "CLEAN | MINOR_ISSUES | MAJOR_ISSUES"
}}
Return ONLY the JSON."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        log = json.loads(raw[start:end])
    except Exception:
        log = {"fallacies_found": [], "suspect_claims": [], "verdict": "CLEAN"}

    log["turn_ref"] = turn_ref
    log["round"] = state["steel_man_round"]

    print(f"  Verdict: {log.get('verdict', 'UNKNOWN')} | Fallacies: {len(log.get('fallacies_found', []))}")

    fact_check_logs = state["fact_check_logs"] + [log]

    # Determine next phase based on round
    round_num = state["steel_man_round"]
    if round_num == 1:
        next_phase = "pro_rebuttal"
    elif round_num == 2:
        next_phase = "pro_rebuttal"
    else:
        next_phase = "synthesize"

    return {**state, "fact_check_logs": fact_check_logs, "phase": next_phase}


# ─────────────────────────────────────────────
#  NODE 5 – PRO REBUTTAL (Steel-Man Protocol)
# ─────────────────────────────────────────────
def pro_rebuttal_node(state: DebateState) -> DebateState:
    round_num = state["steel_man_round"]
    print(f"\n[PRO-AGENT] Steel-Man rebuttal (Round {round_num})...")
    llm = get_llm("pro", temperature=0.75)

    # Find last Con argument
    con_turns = [t for t in state["transcript"] if t["agent"] == "Con"]
    last_con = con_turns[-1]["content"] if con_turns else ""

    system = f"""You ARE this expert — speak in first person:
{state['pro_persona']}

CRITICAL INSTRUCTION — YOUR ASSIGNED POSITION:
The debate topic is: "{state['topic']}"
YOU are the PRO side. You AGREE with and DEFEND this statement as TRUE.
Do NOT abandon or undermine your pro position. Defend it.

You MUST follow the Steel-Man Protocol in exactly 3 labeled sections:

## 1. VALIDATION
State your opponent's single strongest argument accurately and charitably.

## 2. AMPLIFICATION  
Add one piece of logic, data, or nuance that makes their argument even STRONGER.
(This shows intellectual honesty and depth.)

## 3. REFUTATION
Now explain why, despite that strengthened argument, your PRO framework is still superior.
Use specific mechanisms, evidence, or systemic reasoning.

Total length: ~350 words."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""You are defending the position that "{state['topic']}" is TRUE.
Your opponent (CON side) has just argued AGAINST this. Their argument:

{last_con}

Deliver your Steel-Man rebuttal defending the PRO position.""")
    ])

    content = response.content.strip()
    transcript = add_turn(state, "Pro", f"rebuttal_r{round_num}", content)

    print(f"  → {content[:120]}...")
    return {**state, "transcript": transcript, "phase": "con_rebuttal"}


# ─────────────────────────────────────────────
#  NODE 6 – CON REBUTTAL (Steel-Man Protocol)
# ─────────────────────────────────────────────
def con_rebuttal_node(state: DebateState) -> DebateState:
    round_num = state["steel_man_round"]
    print(f"\n[CON-AGENT] Steel-Man rebuttal (Round {round_num})...")
    llm = get_llm("con", temperature=0.75)

    # Find last Pro argument
    pro_turns = [t for t in state["transcript"] if t["agent"] == "Pro"]
    last_pro = pro_turns[-1]["content"] if pro_turns else ""

    system = f"""You ARE this expert — speak in first person:
{state['con_persona']}

CRITICAL INSTRUCTION — YOUR ASSIGNED POSITION:
The debate topic is: "{state['topic']}"
YOU are the CON side. You DISAGREE with this statement and argue it is FALSE or WRONG.
Do NOT agree with the pro side. Do NOT argue in favour of the topic statement.

You MUST follow the Steel-Man Protocol in exactly 3 labeled sections:

## 1. VALIDATION
State your opponent's single strongest argument accurately and charitably.

## 2. AMPLIFICATION  
Add one piece of logic, data, or nuance that makes their argument even STRONGER.
(This shows intellectual honesty and depth.)

## 3. REFUTATION
Now explain why, despite that strengthened argument, your CON framework is still superior.
Use specific mechanisms, evidence, or systemic reasoning.

Total length: ~350 words."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""You are defending the position that "{state['topic']}" is FALSE or WRONG.
Your opponent (PRO side) has just argued IN FAVOUR of this statement. Their argument:

{last_pro}

Deliver your Steel-Man rebuttal defending the CON position.""")
    ])

    content = response.content.strip()
    transcript = add_turn(state, "Con", f"rebuttal_r{round_num}", content)

    print(f"  → {content[:120]}...")

    new_round = round_num + 1
    if new_round <= 2:
        next_phase = "fact_check"
    else:
        next_phase = "synthesize"

    return {**state, "transcript": transcript, "steel_man_round": new_round, "phase": next_phase}


# ─────────────────────────────────────────────
#  NODE 7 – SYNTHESIZER (Hegelian Synthesis)
# ─────────────────────────────────────────────
def synthesizer_node(state: DebateState) -> DebateState:
    print("\n[SYNTHESIZER] Performing Hegelian synthesis...")
    llm = get_llm("synthesizer", temperature=0.5)

    full_text = full_transcript_text(state)
    fc_summary = json.dumps(state["fact_check_logs"], indent=2)

    prompt = f"""You are a master philosopher performing a Hegelian Dialectic Synthesis.

TOPIC: {state['topic']}

FULL DEBATE TRANSCRIPT:
{full_text}

FACT-CHECK AUDIT:
{fc_summary}

Your task — do NOT declare a winner. Instead produce a rigorous synthesis:

# DIALECTIC SYNTHESIS REPORT

## THESIS — Strongest Pro Arguments
List the 3-5 most intellectually robust points from the Pro side.
For each: state the claim, its strongest evidence, and why it endures.

## ANTITHESIS — Strongest Con Arguments  
List the 3-5 most intellectually robust points from the Con side.
For each: state the claim, its strongest evidence, and why it endures.

## SYNTHESIS — The New Integrated Framework
Create a "Version 2.0" model that:
- Integrates the core truths from both sides
- Resolves their apparent contradictions
- Identifies conditions under which each side's logic holds
- Proposes a nuanced position that transcends the binary

## UNRESOLVED TENSIONS
What genuine disagreements remain even after synthesis?
What would need to be empirically proven to resolve them?

## META-OBSERVATION
One paragraph on what this debate reveals about how humans reason about this topic.

Write in authoritative, academic prose. Aim for ~600 words."""

    response = llm.invoke([HumanMessage(content=prompt)])
    synthesis = response.content.strip()

    print(f"  → Synthesis complete ({len(synthesis)} chars)")
    return {**state, "final_synthesis": synthesis, "phase": "done"}


# ─────────────────────────────────────────────
#  ROUTING LOGIC
# ─────────────────────────────────────────────
def route_phase(state: DebateState) -> str:
    phase_map = {
        "orchestrate":   "orchestrator",
        "pro_opening":   "pro_opening",
        "con_opening":   "con_opening",
        "fact_check":    "fact_checker",
        "pro_rebuttal":  "pro_rebuttal",
        "con_rebuttal":  "con_rebuttal",
        "synthesize":    "synthesizer",
        "done":          END,
    }
    return phase_map.get(state["phase"], END)


# ─────────────────────────────────────────────
#  BUILD THE LANGGRAPH
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    builder = StateGraph(DebateState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("pro_opening",  pro_opening_node)
    builder.add_node("con_opening",  con_opening_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("pro_rebuttal", pro_rebuttal_node)
    builder.add_node("con_rebuttal", con_rebuttal_node)
    builder.add_node("synthesizer",  synthesizer_node)

    builder.set_entry_point("orchestrator")

    # Linear routing through phase field
    for node in ["orchestrator", "pro_opening", "con_opening",
                 "fact_checker", "pro_rebuttal", "con_rebuttal", "synthesizer"]:
        builder.add_conditional_edges(node, route_phase)

    return builder.compile()


# ─────────────────────────────────────────────
#  REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_report(final_state: DebateState) -> str:
    divider = "═" * 72
    thin = "─" * 72

    lines = [
        divider,
        "  MULTI-AGENT DIALECTIC SYSTEM (MADS) — DEBATE REPORT",
        divider,
        f"  Thread ID : {final_state['thread_id']}",
        f"  Topic     : {final_state['topic']}",
        f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Models    : PRO={MODELS['pro']} | CON={MODELS['con']}",
        divider,
        "",
        "  PRO PERSONA",
        thin,
        textwrap.fill(final_state["pro_persona"], width=70, initial_indent="  ", subsequent_indent="  "),
        "",
        "  CON PERSONA",
        thin,
        textwrap.fill(final_state["con_persona"], width=70, initial_indent="  ", subsequent_indent="  "),
        "",
        divider,
        "  FULL TRANSCRIPT",
        divider,
    ]

    for t in final_state["transcript"]:
        lines += [
            "",
            f"  ┌─ Turn {t['turn']} │ {t['agent'].upper()} │ Phase: {t['phase']} ─{'─'*20}",
            "",
        ]
        wrapped = textwrap.fill(t["content"], width=68, initial_indent="  │ ", subsequent_indent="  │ ")
        lines.append(wrapped)
        lines.append("  └" + "─" * 60)

    lines += [
        "",
        divider,
        "  FACT-CHECK AUDIT LOG",
        divider,
    ]
    for log in final_state["fact_check_logs"]:
        lines += [
            f"  Round {log.get('round', '?')} | Turns up to #{log.get('turn_ref', '?')} | Verdict: {log.get('verdict', 'N/A')}",
            f"  Fallacies : {log.get('fallacies_found', [])}",
            f"  Suspect   : {log.get('suspect_claims', [])}",
            thin,
        ]

    lines += [
        "",
        divider,
        "  HEGELIAN SYNTHESIS",
        divider,
        "",
    ]
    for para in final_state["final_synthesis"].split("\n"):
        if para.startswith("#"):
            lines.append(f"\n  {para}")
        else:
            lines.append(textwrap.fill(para, width=68, initial_indent="  ", subsequent_indent="  ") if para.strip() else "")

    lines += ["", divider, "  END OF REPORT", divider]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  MAIN ENTRYPOINT
# ─────────────────────────────────────────────
def run_debate(topic: str) -> DebateState:
    print(f"\n{'═'*60}")
    print(f"  MADS — Multi-Agent Dialectic System")
    print(f"  Topic: {topic}")
    print(f"{'═'*60}")

    initial_state: DebateState = {
        "thread_id": str(uuid.uuid4()),
        "topic": topic,
        "pro_persona": "",
        "con_persona": "",
        "transcript": [],
        "fact_check_logs": [],
        "steel_man_round": 0,
        "phase": "orchestrate",
        "final_synthesis": "",
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # Save report
    report = generate_report(final_state)
    report_path = f"outputs/mads_report_{final_state['thread_id'][:8]}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save JSON state
    json_path = f"outputs/mads_state_{final_state['thread_id'][:8]}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, default=str)

    print(f"\n{'═'*60}")
    print(f"  Report saved → {report_path}")
    print(f"  JSON state  → {json_path}")
    print(f"{'═'*60}\n")

    return final_state


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Universal Basic Income should replace all welfare programs"
    run_debate(topic)
