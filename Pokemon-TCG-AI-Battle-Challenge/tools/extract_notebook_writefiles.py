"""Extract %%writefile cells from downloaded Kaggle notebooks for local audit."""
from __future__ import annotations

import argparse
import ast
import base64
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source_path = Path(args.notebook)
    root = Path(args.out)
    count = 0
    if source_path.suffix == ".py":
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        strings = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    strings[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        for variable, filename in (("MAIN_SOURCE", "main.py"), ("DECK_SOURCE", "deck.csv"),
                                   ("GROUP_SOURCE", "group.txt")):
            if isinstance(strings.get(variable), str):
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(strings[variable], encoding="utf-8")
                print(target)
                count += 1
        print("writefiles", count)
        return

    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    literal_strings = {}
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        lines = source.splitlines(keepends=True)
        parse_source = "".join(lines[1:]) if lines and lines[0].strip().startswith("%%") else source
        try:
            tree = ast.parse(parse_source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, str):
                        literal_strings[node.targets[0].id] = value
                    elif node.targets[0].id in {"DECK", "EXPECTED_DECK"} and isinstance(value, list):
                        literal_strings["__deck_list__"] = "\n".join(map(str, value)) + "\n"
                    elif node.targets[0].id in {"PAYLOADS", "PAYLOAD_B64"} and isinstance(value, dict):
                        for filename, payload in value.items():
                            if isinstance(filename, str) and isinstance(payload, str):
                                target = root / filename
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(base64.b64decode(payload))
                                print(target)
                                count += 1
        except SyntaxError:
            pass
        if not lines or not lines[0].strip().startswith("%%writefile "):
            continue
        relative = lines[0].strip().split(None, 1)[1]
        relative_path = Path(relative)
        if relative_path.is_absolute():
            parts = relative_path.parts
            relative_path = Path(*parts[3:]) if len(parts) > 3 and parts[1:3] == ("kaggle", "working") else Path(relative_path.name)
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[1:]), encoding="utf-8")
        print(target)
        count += 1
    if not (root / "main.py").exists() and "MAIN_B64" in literal_strings:
        target = root / "main.py"
        target.write_bytes(base64.b64decode(literal_strings["MAIN_B64"]))
        print(target)
        count += 1
    for names, filename in ((('DECK_SOURCE', 'DECK_TEXT', 'deck_text', '__deck_list__'), 'deck.csv'),
                            (('MAIN_SOURCE', 'agent_source'), 'main.py')):
        if (root / filename).exists():
            continue
        value = next((literal_strings[name] for name in names if name in literal_strings), None)
        if value is not None:
            target = root / filename
            target.write_text(value, encoding="utf-8")
            print(target)
            count += 1
    print("writefiles", count)


if __name__ == "__main__":
    main()
