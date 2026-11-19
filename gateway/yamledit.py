"""YAML-safe, comment-preserving edits for config.yaml (accounts) and models.yaml (model entries).

Neither file is rewritten wholesale: only the lines of the thing being changed are touched, every other
comment and layout stays. Each edit is verified by re-parsing the result and comparing it with the
intended data (everything else must be unchanged); on any mismatch nothing is written. Writes are
atomic (tmp + rename) and keep the previous version next to the file as `<name>.bak`.
"""
import json, os, re, shutil, textwrap
import yaml


class EditError(ValueError):
    """The edit was refused (unknown name, invalid value, would corrupt the file)."""


# ---- plumbing ---------------------------------------------------------------------------------------
def atomic_write(path, text, backup=True):
    path = str(path)
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise EditError(f"refusing to write invalid YAML to {os.path.basename(path)}: {e}") from None
    if backup and os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
        try:
            os.chmod(path + ".bak", os.stat(path).st_mode & 0o777)
        except OSError:
            pass
    tmp = f"{path}.tmp{os.getpid()}"
    mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o600
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _lines(text):
    return text.split("\n")


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def scalar(v):
    """One YAML-safe inline scalar."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)      # a double-quoted JSON string is valid YAML


def flow(v):
    """Inline flow YAML for lists/dicts/scalars (JSON subset: always valid YAML)."""
    return json.dumps(v, ensure_ascii=False)


def _wrapped_list(items, head, cont_indent, width=100):
    """A flow list of strings wrapped over several lines, like models.yaml writes them by hand."""
    toks = [json.dumps(str(x), ensure_ascii=False) for x in items]
    out, cur = [], head + " ["
    for i, tok in enumerate(toks):
        piece = tok + ("]" if i == len(toks) - 1 else ",")
        sep = "" if cur.endswith("[") else " "
        if len(cur) + len(sep) + len(piece) > width and not cur.endswith("["):
            out.append(cur)
            cur = " " * cont_indent + piece
        else:
            cur += sep + piece
    if not toks:
        cur += "]"
    out.append(cur)
    return out


# ---- accounts in config.yaml --------------------------------------------------------------------------
_ITEM = re.compile(r"^(\s*)-\s")


def _accounts_span(lines):
    """(start, end) line indexes of the body of the top-level `accounts:` list."""
    for i, ln in enumerate(lines):
        if re.match(r"^accounts:\s*(#.*)?$", ln):
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith((" ", "\t", "-"))
                                      or lines[j].lstrip().startswith("#") and lines[j].startswith(" ")):
                j += 1
            # do not swallow trailing blank lines / col-0 comments
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            return i, i + 1, j
        if re.match(r"^accounts:\s*(?:\[\s*\]|null|~)\s*(#.*)?$", ln):
            return i, i + 1, i + 1
    return None


def _account_items(lines, a, b):
    """[(start, end, indent)] for each list item between a and b."""
    starts = [i for i in range(a, b) if _ITEM.match(lines[i])]
    items = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else b
        while e > s + 1 and (not lines[e - 1].strip() or lines[e - 1].lstrip().startswith("#")) :
            e -= 1
        items.append((s, e, _indent(lines[s])))
    return items


def _item_data(lines, s, e):
    try:
        d = yaml.safe_load("\n".join(lines[s:e]))
        return d[0] if isinstance(d, list) and d and isinstance(d[0], dict) else {}
    except yaml.YAMLError:
        return {}


def _accounts_rewrite(text, accounts):
    """Fallback: rewrite the whole accounts section from data (comments inside it are lost)."""
    lines = _lines(text)
    span = _accounts_span(lines)
    body = [f"  - {yaml.safe_dump(a, default_flow_style=True, width=10**6, sort_keys=False).strip()}" for a in accounts]
    if span is None:
        if not body:
            return text.rstrip("\n") + "\n\naccounts: []\n"
        return text.rstrip("\n") + "\n\naccounts:\n" + "\n".join(body) + "\n"
    _, a, b = span
    # `accounts: []` / `null` / `~` (the starter config) or any inline value becomes a plain block header; a comment is kept
    if not body:
        head = re.sub(r"^(accounts:)\s*(?:\[[^\]]*\]|\{[^}]*\}|null|~)?\s*(#.*)?$",
                      lambda m: f"{m.group(1)} []" + (f"  {m.group(2)}" if m.group(2) else ""), lines[span[0]])
    else:
        head = re.sub(r"^(accounts:)\s*(?:\[[^\]]*\]|\{[^}]*\}|null|~)?\s*(#.*)?$",
                      lambda m: m.group(1) + (f"  {m.group(2)}" if m.group(2) else ""), lines[span[0]])
    return "\n".join(lines[:span[0]] + [head] + body + lines[b:])


def _check_accounts(new_text, expected):
    try:
        got = (yaml.safe_load(new_text) or {}).get("accounts") or []
    except yaml.YAMLError:
        return False
    return got == expected


def edit_account(text, name, set_fields=None, drop_fields=()):
    """Return config text with account `name` updated (set/drop keys). Raises EditError if unknown."""
    cfg = yaml.safe_load(text) or {}
    accounts = cfg.get("accounts") or []
    idx = next((i for i, a in enumerate(accounts) if a.get("name") == name), None)
    if idx is None:
        raise EditError(f"no account named '{name}'")
    want = dict(accounts[idx])
    for k, v in (set_fields or {}).items():
        want[k] = v
    for k in drop_fields:
        want.pop(k, None)
    expected = [want if i == idx else a for i, a in enumerate(accounts)]
    lines = _lines(text)
    span = _accounts_span(lines)
    if span:
        _, a, b = span
        items = _account_items(lines, a, b)
        if len(items) == len(accounts):
            s, e, ind = items[idx]
            first = lines[s]
            if re.match(r"^\s*-\s*\{", first) and e == s + 1:
                m = re.match(r"^(\s*-\s*)\{.*\}(\s*#.*)?\s*$", first)
                if m:
                    flow_txt = yaml.safe_dump(want, default_flow_style=True, width=10**6, sort_keys=False).strip()
                    lines[s] = f"{m.group(1)}{flow_txt}{m.group(2) or ''}"
                    new = "\n".join(lines)
                    if _check_accounts(new, expected):
                        return new
            else:                                    # block mapping item
                new_lines = list(lines[s:e])
                pad = " " * (ind + 2)
                for k, v in list((set_fields or {}).items()) + [(k, ...) for k in drop_fields]:
                    pat = re.compile(rf"^\s*(?:-\s+)?{re.escape(k)}\s*:")
                    hit = next((i for i, ln in enumerate(new_lines) if pat.match(ln)), None)
                    if v is ...:
                        if hit is not None:
                            if hit == 0 and _ITEM.match(new_lines[0]):
                                break                # dropping the key on the dash line: use fallback
                            del new_lines[hit]
                    elif hit is not None:
                        pre = re.match(r"^(\s*(?:-\s+)?)", new_lines[hit]).group(1)
                        new_lines[hit] = f"{pre}{k}: {scalar(v)}"
                    else:
                        new_lines.append(f"{pad}{k}: {scalar(v)}")
                else:
                    new = "\n".join(lines[:s] + new_lines + lines[e:])
                    if _check_accounts(new, expected):
                        return new
    new = _accounts_rewrite(text, expected)
    if not _check_accounts(new, expected):
        raise EditError("could not edit the accounts list safely")
    return new


def remove_account(text, name):
    cfg = yaml.safe_load(text) or {}
    accounts = cfg.get("accounts") or []
    if not any(a.get("name") == name for a in accounts):
        raise EditError(f"no account named '{name}'")
    expected = [a for a in accounts if a.get("name") != name]
    lines = _lines(text)
    span = _accounts_span(lines)
    if span:
        _, a, b = span
        items = _account_items(lines, a, b)
        if len(items) == len(accounts):
            idx = next(i for i, x in enumerate(accounts) if x.get("name") == name)
            s, e, _ = items[idx]
            new = "\n".join(lines[:s] + lines[e:])
            if _check_accounts(new, expected):
                return new
    return _accounts_rewrite(text, expected)


def append_account(text, entry):
    """Add an account (dict) at the end of the accounts list, keeping everything else."""
    cfg = yaml.safe_load(text) or {}
    expected = list(cfg.get("accounts") or []) + [entry]
    flow_txt = yaml.safe_dump(entry, default_flow_style=True, width=10**6, sort_keys=False).strip()
    lines = _lines(text)
    span = _accounts_span(lines)
    if span:
        _, a, b = span
        items = _account_items(lines, a, b)
        if items:
            s, e, ind = items[-1]
            new = "\n".join(lines[:e] + [f"{' ' * ind}- {flow_txt}"] + lines[e:])
            if _check_accounts(new, expected):
                return new
    new = _accounts_rewrite(text, expected)
    if not _check_accounts(new, expected):
        raise EditError("could not add the account safely")
    return new


# ---- model entries in models.yaml ----------------------------------------------------------------------
def _model_block(lines, key):
    """(start, end) of `  key:` and its fields (indent > 2), excluding trailing blanks/comments."""
    pat = re.compile(rf"^  {re.escape(key)}\s*:\s*(#.*)?$")
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        raise EditError(f"no model named '{key}' in models.yaml")
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and _indent(ln) <= 2 and not ln.startswith(" " * 4):
            break
        end += 1
    while end > start + 1 and (not lines[end - 1].strip() or _indent(lines[end - 1]) <= 2):
        end -= 1
    return start, end


def _fields(lines, start, end):
    """{field: (first_line, last_line_exclusive)} for the indent-4 fields of a block."""
    heads = []
    for i in range(start + 1, end):
        m = re.match(r"^    ([A-Za-z_][\w.-]*)\s*:", lines[i])
        if m:
            heads.append((m.group(1), i))
    out = {}
    for n, (k, i) in enumerate(heads):
        e = heads[n + 1][1] if n + 1 < len(heads) else end
        while e > i + 1 and (not lines[e - 1].strip() or lines[e - 1].lstrip().startswith("#")
                             and _indent(lines[e - 1]) <= 4):
            e -= 1
        out[k] = (i, e)
    return out


def _render_field(k, v, old_line=None):
    """Lines for `    k: value`; the trailing comment of a replaced one-line value is kept."""
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return _wrapped_list(v, f"    {k}:", 11 if k == "args" else 10)
    if isinstance(v, (list, dict)):
        return [f"    {k}: {flow(v)}"]
    plain = isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_./@+-]+", v) and not re.fullmatch(r"[0-9.+-]+", v)
    core = f"    {k}: {v if plain else scalar(v)}"
    m = re.search(r"\s#\s.*$", old_line or "")
    if m and not re.search(r"""["'][^"']*#""", old_line):
        comment = m.group(0).strip()
        return [(core.ljust(42) if len(core) < 42 else core) + "  " + comment]
    return [core]


def update_model(text, key, changes):
    """Return models.yaml text with `changes` ({field: value|None}) applied to model `key`.
    None removes the field. Unchanged fields (and their comments) are not touched."""
    models = (yaml.safe_load(text) or {}).get("models") or {}
    if key not in models:
        raise EditError(f"no model named '{key}' in models.yaml")
    want = dict(models[key])
    for k, v in changes.items():
        if v is None:
            want.pop(k, None)
        else:
            want[k] = v
    if want == models[key]:
        return text
    lines = _lines(text)
    start, end = _model_block(lines, key)
    fields = _fields(lines, start, end)
    # apply bottom-up so earlier indexes stay valid
    edits = []
    for k, v in changes.items():
        if models[key].get(k) == v or (v is None and k not in models[key]):
            continue
        if k in fields:
            s, e = fields[k]
            edits.append((s, e, [] if v is None else _render_field(k, v, lines[s] if e == s + 1 else None)))
        elif v is not None:
            at = max((e for _, e in fields.values()), default=start + 1)
            edits.append((at, at, _render_field(k, v)))
    for s, e, new in sorted(edits, key=lambda x: -x[0]):
        lines[s:e] = new
    out = "\n".join(lines)
    _verify_models(out, {**models, key: want})
    return out


def delete_model(text, key):
    models = (yaml.safe_load(text) or {}).get("models") or {}
    if key not in models:
        raise EditError(f"no model named '{key}' in models.yaml")
    lines = _lines(text)
    start, end = _model_block(lines, key)
    if end < len(lines) and not lines[end].strip():
        end += 1                                   # the blank separator after it
    out = "\n".join(lines[:start] + lines[end:])
    expected = {k: v for k, v in models.items() if k != key}
    _verify_models(out, expected)
    return out


def _verify_models(text, expected):
    try:
        got = (yaml.safe_load(text) or {}).get("models") or {}
    except yaml.YAMLError as e:
        raise EditError(f"edit would make models.yaml invalid ({e}); nothing written") from None
    if got != expected:
        bad = sorted(k for k in set(got) | set(expected) if got.get(k) != expected.get(k))
        raise EditError(f"edit did not round-trip for {', '.join(bad)}; nothing written")
