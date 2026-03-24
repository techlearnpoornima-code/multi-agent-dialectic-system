"""
MADS CLI — Interactive launcher for Multi-Agent Dialectic System
"""

import sys
import subprocess
import importlib.util

def check_ollama():
    """Verify Ollama is running and list available models."""
    try:
        import ollama
        models = ollama.list()
        available = [m.model for m in models.models]
        return available
    except Exception as e:
        print(str(e))
        return None


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       MADS — Multi-Agent Dialectic System v1.0              ║
║       Hegelian Debate Engine · LangGraph + Ollama           ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Check Ollama connectivity
    print("► Checking Ollama connection...")
    available_models = check_ollama()

    if available_models is None:
        print("""
  ✗ Cannot connect to Ollama at http://localhost:11434

  To fix this, run:
    ollama serve

  Then pull models (minimum 1 model required):
    ollama pull llama3.2
    ollama pull mistral
    ollama pull qwen2.5
""")
        sys.exit(1)

    print(f"  ✓ Ollama connected. Available models: {available_models}\n")

    # Show model config
    from mads import MODELS
    print("► Configured model roles:")
    for role, model in MODELS.items():
        status = "✓" if model in available_models else "✗ (NOT FOUND)"
        print(f"  {role:<15} → {model} {status}")

    missing = [m for m in MODELS.values() if m not in available_models]
    if missing:
        print(f"""
  ⚠ Missing models: {missing}
  Run:  ollama pull {" ".join(set(missing))}
  Or edit MODELS dict in mads.py to use available models.
""")
        if input("  Continue anyway? (y/N): ").strip().lower() != "y":
            sys.exit(0)

    print()

    # Get topic
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        print("► Enter debate topic (or press Enter for default):")
        print("  Example topics:")
        print("  • AI will cause more harm than good to society")
        print("  • Universal Basic Income should replace welfare programs")
        print("  • Social media companies should be liable for user content")
        print()
        topic = input("  Topic: ").strip()
        if not topic:
            topic = "Artificial Intelligence will cause more harm than good to society"
            print(f"  Using default: {topic}")

    print()
    from mads import run_debate
    final_state = run_debate(topic)

    # Print synthesis preview
    print("\n" + "═"*60)
    print("  SYNTHESIS PREVIEW (first 500 chars)")
    print("═"*60)
    print(final_state["final_synthesis"][:500] + "...")
    print("═"*60)
    print()


if __name__ == "__main__":
    main()
