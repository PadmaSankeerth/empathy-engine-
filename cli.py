#!/usr/bin/env python3
"""
Empathy Engine — CLI Interface

Usage:
  python cli.py "I am so happy today!"
  python cli.py --text "I can't believe this happened." --output output.mp3
  python cli.py --interactive
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from emotion_engine import EmotionEngine
from ssml_builder import SSMLBuilder
from tts_engine import TTSEngine
from voice_mapper import VoiceMapper


EMOTION_COLORS = {
    "joy":      "\033[93m",   # yellow
    "anger":    "\033[91m",   # red
    "sadness":  "\033[94m",   # blue
    "fear":     "\033[95m",   # magenta
    "surprise": "\033[33m",   # orange-ish
    "disgust":  "\033[92m",   # green
    "neutral":  "\033[37m",   # gray
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def print_banner():
    print(f"""
{BOLD}╔══════════════════════════════════════╗
║       THE EMPATHY ENGINE  v1.0       ║
║   AI Voice with Emotional Resonance  ║
╚══════════════════════════════════════╝{RESET}
""")


def run_pipeline(text: str, output: str, engine, mapper, builder, tts):
    """Run the full pipeline and return output path."""
    print(f"\n{DIM}Analyzing:{RESET} {text[:80]}{'...' if len(text) > 80 else ''}\n")

    t0 = time.time()

    # Step 1: Detect emotion
    result = engine.analyze(text)
    color  = EMOTION_COLORS.get(result.label, RESET)
    pct    = int(result.intensity * 100)

    print(f"  Emotion    {color}{BOLD}{result.label.upper()}{RESET}   (confidence: {result.confidence:.0%})")
    print(f"  Intensity  {'█' * (pct // 10)}{'░' * (10 - pct // 10)}  {pct}%")
    print(f"  VADER      compound={result.compound:+.3f}")

    if result.all_scores:
        top3 = sorted(result.all_scores.items(), key=lambda x: -x[1])[:3]
        scores_str = "  ".join(f"{k}:{v:.0%}" for k,v in top3)
        print(f"  Top-3      {DIM}{scores_str}{RESET}")

    # Step 2: Map to voice params
    params = mapper.map(result.label, result.intensity)
    print(f"\n  Rate       {BOLD}{params.rate}{RESET}")
    print(f"  Pitch      {BOLD}{params.pitch}{RESET}")
    print(f"  Volume     {BOLD}{params.volume}{RESET}")
    print(f"  Break      {BOLD}{params.break_ms}ms{RESET}")
    print(f"  Emphasis   {BOLD}{params.emphasis}{RESET}")

    # Step 3: Build SSML
    ssml = builder.build(text, params)
    print(f"\n  SSML       {DIM}{ssml[:100]}{'...' if len(ssml) > 100 else ''}{RESET}")

    # Step 4: Synthesize
    print(f"\n  Synthesizing audio...")
    success = tts.synthesize(ssml, output, params)

    elapsed = int((time.time() - t0) * 1000)
    if success:
        print(f"\n  {BOLD}✓ Audio saved:{RESET} {output}  ({elapsed}ms)")
    else:
        print(f"\n  {BOLD}✗ Synthesis failed.{RESET}")

    return success


def interactive_mode(engine, mapper, builder, tts):
    print(f"{DIM}Interactive mode — type text and press Enter. Type 'quit' to exit.{RESET}\n")
    counter = 1
    while True:
        try:
            text = input(f"[{counter}] Text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not text:
            continue

        output = f"output_{counter:03d}.mp3"
        run_pipeline(text, output, engine, mapper, builder, tts)
        counter += 1
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Empathy Engine CLI — Emotionally expressive TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "I am so happy today!"
  python cli.py --text "This is terrible." --output sad.mp3
  python cli.py --interactive
  python cli.py --health
        """
    )
    parser.add_argument("text", nargs="?", help="Text to synthesize (positional)")
    parser.add_argument("--text", "-t", dest="text_flag", help="Text to synthesize (flag)")
    parser.add_argument("--output", "-o", default="output.mp3", help="Output MP3 path (default: output.mp3)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--health", action="store_true", help="Print subsystem health and exit")
    args = parser.parse_args()

    print_banner()

    # Load subsystems
    print(f"{DIM}Loading models...{RESET}", end="", flush=True)
    engine  = EmotionEngine()
    mapper  = VoiceMapper()
    builder = SSMLBuilder()
    tts     = TTSEngine()
    print(f" {BOLD}ready{RESET}")

    # Health check
    if args.health:
        em  = engine.health_check()
        tts_h = tts.health_check()
        print(f"\nHealth:")
        print(f"  HuggingFace  {'✓' if em['huggingface'] else '✗'}")
        print(f"  VADER        {'✓' if em['vader'] else '✗'}")
        print(f"  TTS backend  {tts_h['backend']}")
        print(f"  SSML support {'✓' if tts_h['supports_ssml'] else '✗ (post-process only)'}")
        return

    # Determine text input
    text = args.text_flag or args.text

    if args.interactive:
        interactive_mode(engine, mapper, builder, tts)
    elif text:
        success = run_pipeline(text, args.output, engine, mapper, builder, tts)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
