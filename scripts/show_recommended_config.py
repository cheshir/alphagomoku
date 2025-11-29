#!/usr/bin/env python
"""Show recommended training configuration for your hardware.

Usage:
    python scripts/show_recommended_config.py
    python scripts/show_recommended_config.py --prefer-speed
    python scripts/show_recommended_config.py --prefer-strength
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphagomoku.utils.hardware_config import (
    detect_hardware,
    get_recommended_config,
    print_hardware_info,
    print_recommended_config,
    check_memory_sufficient,
)


def main():
    parser = argparse.ArgumentParser(
        description="Show recommended training configuration for your hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show default recommendation (balanced)
  python scripts/show_recommended_config.py

  # Optimize for training speed
  python scripts/show_recommended_config.py --prefer-speed

  # Optimize for model strength
  python scripts/show_recommended_config.py --prefer-strength

The script will:
  1. Detect your hardware (CPU, RAM, GPU)
  2. Recommend optimal training settings
  3. Show the command to run
        """,
    )

    parser.add_argument(
        "--prefer-speed",
        action="store_true",
        help="Optimize for training speed (smaller model, more workers)",
    )
    parser.add_argument(
        "--prefer-strength",
        action="store_true",
        help="Optimize for model strength (larger model, more games)",
    )
    parser.add_argument(
        "--no-command",
        action="store_true",
        help="Don't show the command to run",
    )

    args = parser.parse_args()

    # Can't prefer both
    if args.prefer_speed and args.prefer_strength:
        print("❌ Error: Cannot use both --prefer-speed and --prefer-strength")
        sys.exit(1)

    # Detect hardware
    print("\n🔍 Detecting hardware...\n")
    hardware = detect_hardware()
    print_hardware_info(hardware)

    # Get recommendations
    print("\n")
    if args.prefer_speed:
        print("⚡ CONFIGURATION: Optimized for SPEED")
        config = get_recommended_config(hardware, prefer_speed=True)
    elif args.prefer_strength:
        print("💪 CONFIGURATION: Optimized for STRENGTH")
        config = get_recommended_config(hardware, prefer_strength=True)
    else:
        print("⚖️  CONFIGURATION: Balanced (speed + strength)")
        config = get_recommended_config(hardware)

    print_recommended_config(config, show_command=not args.no_command)

    # Memory check
    print("\n")
    sufficient, msg = check_memory_sufficient(config, hardware)
    print(msg)

    if not sufficient:
        print("\n⚠️  WARNING: You may run out of memory with these settings!")
        print("Consider using a smaller model preset or fewer parallel workers.")

    # Show alternatives
    if not args.prefer_speed and not args.prefer_strength:
        print("\n" + "=" * 70)
        print("OTHER OPTIONS")
        print("=" * 70)
        print("\nFor faster iteration (smaller model):")
        print("  python scripts/show_recommended_config.py --prefer-speed")
        print("\nFor stronger model (larger model, more data):")
        print("  python scripts/show_recommended_config.py --prefer-strength")

    print("\n✅ Ready to train! Copy the command above or run: make train\n")


if __name__ == "__main__":
    main()
