"""Generate the pages VitePress cannot build from source alone.

1. Every notebook under ``examples/`` becomes ``docs/examples/<path>.md`` with its saved outputs; images and videos
   are written next to it as files.
3. Every English page without a translation, notebooks and API pages included, in ``docs/<locale>/`` gets an English copy, with a note, in
   ``docs/_fallback/<locale>/``; config.mts rewrites it to ``/<locale>/<path>``.
2. Every ``::: torch_world.<path>`` directive in ``docs/api/*.md`` is rendered from the docstrings with griffe into
   ``docs/.vitepress/api.json``, which the markdown plugin in ``markdown.ts`` splices into the page.

Run with ``npm run docs:prepare``; ``docs:dev`` and ``docs:build`` run it first.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import griffe

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
NOTEBOOKS_OUT = DOCS / "examples"
API_OUT = DOCS / ".vitepress" / "api.json"
REPO = "https://github.com/latent-horizon/pytorch-world"
FALLBACK_OUT = DOCS / "_fallback"

# The note on an untranslated page, per locale (the locales of i18n.ts).
FALLBACK_NOTES = {
    "pt": ("Esta página ainda não foi traduzida.", "Leia em inglês"),
    "es": ("Esta página aún no está traducida.", "Leer en inglés"),
    "zh": ("本页尚未翻译。", "阅读英文版"),
    "ja": ("このページはまだ翻訳されていません。", "英語版を読む"),
}

# Run artifacts and scratch folders that hold notebooks but are not part of the site.
SKIP_DIRS = {"runs", "results", "checkpoints", "reports", "media", "records", ".ipynb_checkpoints"}
MEDIA = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4", ".webm"}
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/svg+xml": "svg", "video/mp4": "mp4"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DATA_URI = re.compile(r'src="data:([\w/+.-]+);base64,([^"]+)"')

# Kept in sync with the regex in markdown.ts.
DIRECTIVE = re.compile(r"^::: (torch_world[\w.]*)[ \t]*\n((?:[ \t]+\S.*\n)*)", re.MULTILINE)


# ---------- notebooks ----------


def _text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def _stream(text: str) -> str:
    # Keep what a terminal would show: the last carriage-return rewrite of each line, no colors.
    lines = [line.rsplit("\r", 1)[-1] for line in ANSI.sub("", text).split("\n")]
    return "\n".join(lines).strip("\n")


def _fence(text: str, lang: str = "text") -> str:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _guard(markdown: str) -> str:
    # Vue would read {{ }} as an interpolation.
    return f"::: v-pre\n{markdown}\n:::" if "{{" in markdown else markdown


class Notebook:
    def __init__(self, source: Path):
        self.source = source
        self.rel = source.relative_to(EXAMPLES)
        self.page = NOTEBOOKS_OUT / self.rel.with_suffix(".md")
        self.files = self.page.parent / f"{source.stem}_files"
        self.count = 0

    def _write(self, data: bytes, ext: str) -> str:
        self.files.mkdir(parents=True, exist_ok=True)
        self.count += 1
        name = f"output_{self.count}.{ext}"
        (self.files / name).write_bytes(data)
        return f"./{self.files.name}/{name}"

    def _copy_local(self, markdown: str) -> str:
        # Relative paths in markdown cells are written for the notebook's place in the repository. Images travel
        # with the page, links into docs/ point at the page, and links to other repository files go to GitHub.
        def relink(match: re.Match) -> str:
            target, _, anchor = match.group(2).partition("#")
            path = (self.source.parent / target).resolve()
            if not target or not path.is_relative_to(ROOT):
                return match.group(0)
            if Path(target).suffix.lower() in MEDIA and path.is_file() and path.is_relative_to(EXAMPLES):
                dest = self.page.parent / target
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                return match.group(0)
            if path.is_relative_to(DOCS):
                url = Path(os.path.relpath(path, self.page.parent)).as_posix()
            elif path.is_relative_to(EXAMPLES) and path.suffix == ".ipynb":
                return match.group(0)
            else:
                url = f"{REPO}/blob/main/{path.relative_to(ROOT).as_posix()}"
            return match.group(1) + url + (f"#{anchor}" if anchor else "")

        return re.sub(r"""(\]\(|src=["'])(?!https?:|data:|/|#|mailto:)([^)"'\s]+)""", relink, markdown)

    def _attachments(self, cell: dict, markdown: str) -> str:
        for name, bundle in cell.get("attachments", {}).items():
            for mime, data in bundle.items():
                if mime in EXTENSIONS:
                    url = self._write(base64.b64decode(_text(data)), EXTENSIONS[mime])
                    markdown = markdown.replace(f"attachment:{name}", url)
                    break
        return markdown

    def _html(self, html: str) -> str:
        html = DATA_URI.sub(
            lambda m: f'src="{self._write(base64.b64decode(m.group(2)), EXTENSIONS.get(m.group(1), "bin"))}"', html
        )
        # One HTML block: markdown-it ends a block at the first blank line.
        body = "\n".join(line for line in html.strip().splitlines() if line.strip())
        pre = " v-pre" if "{{" in body else ""
        return f'<div class="nb-output nb-html"{pre}>\n{body}\n</div>'

    def _output(self, output: dict) -> str | None:
        kind = output["output_type"]
        if kind == "stream":
            text = _stream(_text(output["text"]))
            return _fence(text) if text else None
        if kind == "error":
            return _fence(_stream("\n".join(output["traceback"])))
        data = output.get("data", {})
        for mime in ("image/png", "image/jpeg", "image/gif"):
            if mime in data:
                return f"![]({self._write(base64.b64decode(_text(data[mime])), EXTENSIONS[mime])})"
        if "image/svg+xml" in data:
            return f"![]({self._write(_text(data['image/svg+xml']).encode(), 'svg')})"
        if "text/html" in data:
            return self._html(_text(data["text/html"]))
        if "text/markdown" in data:
            return _guard(_text(data["text/markdown"]).strip())
        if "text/latex" in data:
            return _text(data["text/latex"]).strip()
        if "text/plain" in data:
            return _fence(_stream(_text(data["text/plain"])))
        return None

    def render(self) -> None:
        nb = json.loads(self.source.read_text())
        lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
        blocks = [f"---\nnotebook: examples/{self.rel.as_posix()}\n---"]
        for cell in nb["cells"]:
            source = _text(cell["source"]).strip()
            if cell["cell_type"] == "markdown":
                if source:
                    blocks.append(_guard(self._copy_local(self._attachments(cell, source))))
            elif cell["cell_type"] == "code":
                if source:
                    blocks.append(_fence(source, lang))
                outputs = [rendered for o in cell.get("outputs", []) if (rendered := self._output(o))]
                if outputs:
                    blocks.append('<div class="nb-output">\n\n' + "\n\n".join(outputs) + "\n\n</div>")
        self.page.parent.mkdir(parents=True, exist_ok=True)
        self.page.write_text("\n\n".join(blocks) + "\n")


def notebooks() -> int:
    if NOTEBOOKS_OUT.is_symlink():
        NOTEBOOKS_OUT.unlink()
    elif NOTEBOOKS_OUT.is_dir():
        shutil.rmtree(NOTEBOOKS_OUT)
    sources = [p for p in sorted(EXAMPLES.rglob("*.ipynb")) if not SKIP_DIRS & set(p.relative_to(EXAMPLES).parts)]
    for source in sources:
        Notebook(source).render()
    return len(sources)


# ---------- untranslated pages ----------


def _english_pages() -> list[Path]:
    """English pages, including the generated notebooks: everything but the home page and locale or asset folders."""
    skip = set(FALLBACK_NOTES) | {"_fallback", "public", "assets", ".vitepress"}
    pages = (p.relative_to(DOCS) for p in sorted(DOCS.rglob("*.md")))
    return [DOCS / p for p in pages if p.parts[0] not in skip and p.as_posix() != "index.md"]


def _relink(source: Path, dest: Path, markdown: str) -> str:
    """Point the relative links of an English page copied to ``dest`` at the right places."""

    def fix(match: re.Match) -> str:
        target, _, anchor = match.group(2).partition("#")
        resolved = (source.parent / target).resolve()
        if not target or not resolved.is_relative_to(DOCS):
            return match.group(0)
        if resolved.suffix in (".md", ".ipynb") or resolved.is_dir():
            return match.group(0)  # the locale mirrors the English tree
        # Images and videos stay where they are, so each file is published once for every language.
        url = Path(os.path.relpath(resolved, dest.parent)).as_posix()
        return match.group(1) + url + (f"#{anchor}" if anchor else "")

    pattern = r"""(\]\(|(?:src|href)=["'])(?!https?:|data:|/|#|mailto:)([^)"'\s]+)"""
    parts = re.split(r"(^```.*?^```)", markdown, flags=re.MULTILINE | re.DOTALL)
    return "".join(p if p.startswith("```") else re.sub(pattern, fix, p) for p in parts)


def fallbacks() -> int:
    if FALLBACK_OUT.is_dir():
        shutil.rmtree(FALLBACK_OUT)
    count = 0
    for source in _english_pages():
        rel = source.relative_to(DOCS)
        text = source.read_text()
        front = ""
        if text.startswith("---\n"):
            end = text.index("\n---\n", 4) + 5
            front, text = text[:end], text[end:]
        url = "/" + rel.as_posix()
        for locale, (note, english) in FALLBACK_NOTES.items():
            if (DOCS / locale / rel).exists():
                continue
            dest = FALLBACK_OUT / locale / rel
            body = _relink(source, dest, text)
            notice = f"::: info\n{note} [{english}]({url})\n:::\n"
            heading = re.search(r"^# .*\n", body, flags=re.MULTILINE)
            at = heading.end() if heading else 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(front + body[:at] + "\n" + notice + body[at:])
            count += 1
    return count


# ---------- API reference ----------


def _options(block: str) -> dict:
    """The few mkdocstrings options the API pages use: ``members: [a, b]`` and ``show_root_heading: false``."""
    options = {}
    for line in block.splitlines():
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if key == "members":
            options["members"] = [m.strip() for m in value.strip("[]").split(",") if m.strip()]
        elif key == "show_root_heading":
            options["show_root_heading"] = value.lower() == "true"
    return options


def _cell(text: object) -> str:
    return str(text).strip().replace("|", "\\|").replace("\n\n", "<br><br>").replace("\n", " ")


def _code(text: object) -> str:
    text = str(text)
    ticks = "``" if "`" in text else "`"
    pad = " " if "`" in text else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


class Signatures:
    """Collects signatures, then formats them all with one ruff call, like mkdocstrings' ``separate_signature``."""

    def __init__(self):
        self.items: list[str] = []

    def add(self, signature: str) -> str:
        self.items.append(signature)
        return f"\x00SIG{len(self.items) - 1}\x00"

    def resolve(self, text: str) -> str:
        source = "".join(f"# SIG{i}\ndef {s}: pass\n" for i, s in enumerate(self.items))
        try:
            out = subprocess.run(
                ["ruff", "format", "--line-length", "80", "-"], input=source, capture_output=True, text=True, check=True
            ).stdout
            parts = re.split(r"^# SIG\d+\n", out, flags=re.MULTILINE)[1:]
            formatted = [re.sub(r":\s*pass\s*$", "", p.strip())[4:] for p in parts]
        except (OSError, subprocess.CalledProcessError):
            formatted = self.items
        if len(formatted) != len(self.items):
            formatted = self.items
        return re.sub(r"\x00SIG(\d+)\x00", lambda m: formatted[int(m.group(1))], text)


def _parameters(params, drop_self: bool) -> str:
    out, seen_star, prev = [], False, None
    for i, p in enumerate(params):
        if drop_self and i == 0 and p.name in ("self", "cls"):
            continue
        kind = p.kind.value if p.kind else ""
        if prev == "positional-only" and kind != "positional-only":
            out.append("/")
        if kind == "keyword-only" and not seen_star:
            out.append("*")
            seen_star = True
        prefix = "*" if kind == "variadic positional" else "**" if kind == "variadic keyword" else ""
        seen_star = seen_star or kind == "variadic positional"
        text = prefix + p.name
        if p.annotation is not None:
            text += f": {p.annotation}"
        if p.default is not None and not prefix:
            text += f" = {p.default}" if p.annotation is not None else f"={p.default}"
        out.append(text)
        prev = kind
    if prev == "positional-only":
        out.append("/")
    return ", ".join(out)


class Renderer:
    def __init__(self, package: griffe.Module, signatures: Signatures):
        self.package = package
        self.signatures = signatures

    def get(self, path: str) -> griffe.Object:
        obj = self.package if path == self.package.path else self.package[path.removeprefix(self.package.path + ".")]
        return obj.final_target if obj.is_alias else obj

    @staticmethod
    def kind(obj) -> str:
        if obj.is_class:
            return "class"
        if obj.is_module:
            return "module"
        if "property" in obj.labels:
            return "property"
        if obj.is_function:
            return "method" if obj.parent is not None and obj.parent.is_class else "function"
        return "attribute"

    def heading(self, obj, level: int, path: str) -> str:
        kind = self.kind(obj)
        return f'{"#" * level} <span class="VPBadge api-kind api-kind--{kind}">{kind}</span> <code>{obj.name}</code> {{#{path}}}'

    def signature(self, obj) -> str | None:
        kind = self.kind(obj)
        if kind == "class":
            init = obj.all_members.get("__init__")
            params = _parameters(init.parameters, drop_self=True) if init is not None else ""
            sig = f"{obj.name}({params})"
        elif kind in ("function", "method"):
            sig = f"{obj.name}({_parameters(obj.parameters, drop_self=kind == 'method')})"
            if obj.returns is not None:
                sig += f" -> {obj.returns}"
        elif kind in ("attribute", "property"):
            text = obj.name + (f": {obj.annotation}" if obj.annotation is not None else "")
            if kind == "attribute" and obj.value is not None and len(str(obj.value)) < 80:
                text += f" = {obj.value}"
            return f'<div class="api-signature">\n\n```python\n{text}\n```\n\n</div>'
        else:
            return None
        return f'<div class="api-signature">\n\n```python\n{self.signatures.add(sig)}\n```\n\n</div>'

    def sections(self, obj) -> list:
        sections = list(obj.docstring.parsed) if obj.docstring else []
        if obj.is_class:
            init = obj.members.get("__init__")
            if init is not None and not init.is_alias and init.docstring:
                sections += init.docstring.parsed
        return sections

    def docstring(self, obj) -> str:
        out = []
        for section in self.sections(obj):
            kind = section.kind.value
            if kind == "text":
                out.append(section.value)
            elif kind in ("parameters", "other parameters", "attributes"):
                title = section.title or kind.capitalize()
                rows = ["| Name | Type | Description | Default |", "|---|---|---|---|"]
                for p in section.value:
                    default = getattr(p, "default", None)
                    rows.append(
                        f"| {_code(p.name)} | {_code(p.annotation) if p.annotation else ''} | {_cell(p.description)} "
                        f"| {_code(default) if default is not None else '*required*' if kind == 'parameters' else ''} |"
                    )
                out.append(f"{title}\n{{.api-section}}\n\n" + "\n".join(rows))
            elif kind in ("returns", "yields", "receives", "raises", "warns"):
                title = section.title or kind.capitalize()
                rows = ["| Type | Description |", "|---|---|"]
                for item in section.value:
                    name = getattr(item, "name", None)
                    annotation = _code(item.annotation) if item.annotation else ""
                    label = f"{_code(name)}: {annotation}" if name else annotation
                    rows.append(f"| {label} | {_cell(item.description)} |")
                out.append(f"{title}\n{{.api-section}}\n\n" + "\n".join(rows))
            elif kind == "examples":
                body = []
                for sub, text in section.value:
                    body.append(_fence(text, "python") if sub.value == "examples" else text)
                out.append(f"{section.title or 'Examples'}\n{{.api-section}}\n\n" + "\n\n".join(body))
            elif kind == "admonition":
                container = {"warning": "warning", "danger": "danger", "tip": "tip"}.get(section.value.kind, "info")
                out.append(
                    f"::: {container} {section.title or section.value.kind.capitalize()}\n"
                    f"{section.value.description}\n:::"
                )
            elif kind == "deprecated":
                out.append(f"::: warning Deprecated {section.value.version or ''}\n{section.value.description}\n:::")
        return "\n\n".join(out)

    def members(self, obj, names: list[str] | None) -> list:
        if names is not None:
            return [obj.members[n] for n in names if n in obj.members]
        found = []
        for name, member in obj.members.items():
            if name.startswith("_") or member.is_alias and obj.is_module and not member.resolved:
                continue
            target = member.final_target if member.is_alias else member
            if target.is_module or member.is_alias and obj.is_module:
                continue
            if self.documented(target):
                found.append(target)
        return sorted(found, key=lambda m: m.lineno or 0)

    def documented(self, obj) -> bool:
        if obj.docstring:
            return True
        return obj.is_class and any(
            not n.startswith("_") and not m.is_alias and self.documented(m) for n, m in obj.members.items()
        )

    @staticmethod
    def meta(obj, path: str) -> str:
        """The import path, and a link to the object's lines on GitHub."""
        meta = f'<code class="api-path">{path}</code>'
        if obj.lineno and obj.relative_filepath:
            lines = f"#L{obj.lineno}" + (f"-L{obj.endlineno}" if obj.endlineno else "")
            url = f"{REPO}/blob/main/{obj.relative_filepath.as_posix()}{lines}"
            meta += f' <a class="api-github" href="{url}" target="_blank" rel="noopener">GitHub</a>'
        return f'<p class="api-meta">{meta}</p>'

    @staticmethod
    def source(obj) -> str | None:
        """The object's code, folded away under its file name, with the file's own line numbers."""
        if obj.is_module or not obj.lineno or not obj.relative_filepath:
            return None
        try:
            code = obj.source
        except (OSError, ValueError):
            return None
        if not code.strip():
            return None
        return (
            '<details class="api-source">\n<summary>Source code in '
            f"<code>{obj.relative_filepath.as_posix()}</code></summary>\n\n"
            f"{_fence(code.rstrip(), f'python:line-numbers={obj.lineno}')}\n\n</details>"
        )

    def summary(self, members: list, path: str) -> str | None:
        """A table of a class's documented members, each linked to its section."""
        if len(members) < 2:
            return None
        rows = ["| Member | Description |", "|---|---|"]
        for member in members:
            first = member.docstring.value.strip().split("\n\n")[0] if member.docstring else ""
            first = re.split(r"(?<=\.)\s", first.replace("\n", " "), maxsplit=1)[0]
            rows.append(f"| [`{member.name}`](#{path}.{member.name}) | {_cell(first)} |")
        return "Members\n{.api-section}\n\n" + "\n".join(rows)

    def render(self, obj, level: int, path: str, options: dict | None = None) -> str:
        options = options or {}
        out = []
        show_heading = options.get("show_root_heading", True)
        if show_heading:
            out.append(self.heading(obj, level, path))
            out.append(self.meta(obj, path))
            if (sig := self.signature(obj)) is not None:
                out.append(sig)
        if doc := self.docstring(obj):
            out.append(doc)
        members = []
        if obj.is_class or obj.is_module:
            members = [m.final_target if m.is_alias else m for m in self.members(obj, options.get("members"))]
            if obj.is_class and (table := self.summary(members, path)):
                out.append(table)
        if show_heading and (code := self.source(obj)):
            out.append(code)
        child = level + 1 if show_heading else level
        for member in members:
            out.append(self.render(member, child, f"{path}.{member.name}"))
        return "\n\n".join(out)


def api() -> int:
    package = griffe.load(
        "torch_world",
        search_paths=[str(ROOT)],
        docstring_parser="google",
        resolve_aliases=True,
        resolve_implicit=False,
        resolve_external=False,
    )
    signatures = Signatures()
    renderer = Renderer(package, signatures)
    pages: dict[str, list[str]] = {}
    count = 0
    for page in sorted((DOCS / "api").glob("*.md")):
        blocks = []
        for match in DIRECTIVE.finditer(page.read_text()):
            path = match.group(1)
            blocks.append(renderer.render(renderer.get(path), 2, path, _options(match.group(2))))
            count += 1
        if blocks:
            pages[page.relative_to(DOCS).as_posix()] = blocks
    rendered = {page: [signatures.resolve(b) for b in blocks] for page, blocks in pages.items()}
    API_OUT.write_text(json.dumps(rendered, indent=1) + "\n")
    return count


if __name__ == "__main__":
    print(f"notebooks: {notebooks()} pages in {NOTEBOOKS_OUT.relative_to(ROOT)}")
    print(f"fallbacks: {fallbacks()} untranslated pages in {FALLBACK_OUT.relative_to(ROOT)}")
    print(f"api: {api()} objects in {API_OUT.relative_to(ROOT)}")
