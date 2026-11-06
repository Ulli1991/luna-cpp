#!/usr/bin/env python3
"""Fail on a run of more than 3 full-line // comments, unless it sits directly above a function, class, struct, template or enum.

    python3 scripts/check_comments.py [dir ...]     # default: src next to this script's parent
"""
import pathlib
import re
import sys

DECL = re.compile(r"\s*(template\b|struct\b|class\b|enum\b|union\b)|.*\(")
COMMENT = re.compile(r"\s*//")


def check(path):
    lines = path.read_text().split("\n")
    bad = []
    i = 0
    while i < len(lines):
        if not COMMENT.match(lines[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and COMMENT.match(lines[j + 1]):
            j += 1
        below = lines[j + 1] if j + 1 < len(lines) else ""
        if j - i + 1 > 3 and not DECL.match(below):
            bad.append(f"{path}:{i + 1}: {j - i + 1} comment lines in a row (more than 3)")
        i = j + 1
    return bad


def main(argv):
    roots = [pathlib.Path(a) for a in argv] or [pathlib.Path(__file__).resolve().parent.parent / "src"]
    bad = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix in (".cpp", ".hpp", ".cu", ".h"):
                bad += check(path)
    print("\n".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
