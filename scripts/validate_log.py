from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rheon.validation import Issue, validate_xes, validation_passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a rheon XES log.")
    parser.add_argument("paths", nargs="+", help="XES file(s) to validate")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args(argv)

    all_results: dict[str, list[Issue]] = {}
    exit_code = 0
    for path in args.paths:
        issues = validate_xes(path)
        all_results[path] = issues
        if not validation_passed(issues, strict=args.strict):
            exit_code = 1
    if args.json:
        print(json.dumps(all_results, indent=2, sort_keys=True))
    else:
        for path, issues in all_results.items():
            status = "PASS" if validation_passed(issues, strict=args.strict) else "FAIL"
            print(f"{path}: {status}")
            for issue in issues:
                print(f"  [{issue['severity']}] {issue['code']}: {issue['message']} {issue['details']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
