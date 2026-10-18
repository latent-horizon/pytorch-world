#!/usr/bin/env bash
# Render the published GitHub releases as docs/changelog.md, newest first.
# The release workflow writes each release's notes, so this page needs no hand edits.
set -euo pipefail

repo="${GITHUB_REPOSITORY:-latent-horizon/pytorch-world}"
out="${1:-docs/changelog.md}"

{
  printf '# Changelog\n\n'
  printf 'Release notes for every published version, newest first. Each release is also on [GitHub](https://github.com/%s/releases) and [PyPI](https://pypi.org/project/torch-world/#history).\n' "${repo}"
  # Release bodies start at "##"; push their headings one level down under the version heading.
  gh api "repos/${repo}/releases" --paginate --jq '
    .[] | select(.draft | not)
    | "\n## [\(.tag_name)](\(.html_url)) · \(.published_at[:10])\n\n\(.body // "" | gsub("\r"; "") | gsub("(?m)^(?<h>#+) "; "#\(.h) "))\n"'
} > "${out}"
