"""Fail CI on missing coverage, failures, or unexpected skips in pytest JUnit XML."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def check_report(path: Path, minimum: int, maximum_xfails: int = 1) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped, xfailed = [], []
    for case in cases:
        entry = case.find("skipped")
        if entry is not None:
            (xfailed if entry.get("type") == "pytest.xfail" else skipped).append(case)
    passed = len(cases) - len(failed) - len(skipped) - len(xfailed)
    if failed or skipped or len(xfailed) > maximum_xfails or passed < minimum:
        raise ValueError(f"CI test gate failed: passed={passed} minimum={minimum}, "
                         f"failed={len(failed)}, unexpected_skips={len(skipped)}, xfailed={len(xfailed)}")
    return {"passed": passed, "xfailed": len(xfailed), "skipped": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum", type=int, required=True)
    parser.add_argument("--maximum-xfails", type=int, default=1)
    args = parser.parse_args(argv)
    print(check_report(args.report, args.minimum, args.maximum_xfails))


if __name__ == "__main__":
    main()
