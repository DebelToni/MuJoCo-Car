from __future__ import annotations

import argparse
from pathlib import Path

import xacro


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile tank xacro to URDF.")
    parser.add_argument("--xacro", type=Path, default=Path("assets/tank_car.xacro"))
    parser.add_argument("--out", type=Path, default=Path("assets/generated/tank_car.urdf"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = xacro.process_file(str(args.xacro))
    xml_text = doc.toprettyxml(indent="  ")
    args.out.write_text(xml_text, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
