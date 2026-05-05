import json
from pathlib import Path


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), None
            except json.JSONDecodeError as exc:
                yield None, "%s:%d: %s" % (path, line_no, exc)


def _scan_worker_dirs_in_bases(bases, prefix):
    found = {}
    for base in bases:
        if base.is_dir() and base.name.startswith(prefix):
            found[str(base.resolve())] = base
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                found[str(child.resolve())] = child
    return found


def find_worker_dirs(trace_root, model, build_tag):
    root = Path(trace_root).expanduser()
    prefix = "%s-a" % model
    bases = []
    if build_tag:
        tagged = root / build_tag
        if tagged.exists():
            bases.append(tagged)
        else:
            bases.append(root)
    else:
        bases.append(root)

    found = _scan_worker_dirs_in_bases(bases, prefix)

    if not found and build_tag and root.is_dir():
        found.update(_scan_worker_dirs_in_bases([root], prefix))

    if not found and not build_tag and root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith(prefix):
                continue
            for grandchild in child.iterdir():
                if grandchild.is_dir() and grandchild.name.startswith(prefix):
                    found[str(grandchild.resolve())] = grandchild

    return [found[key] for key in sorted(found)]
