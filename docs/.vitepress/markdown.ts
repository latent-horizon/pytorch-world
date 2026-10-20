// Markdown extensions that keep the pages close to plain Markdown:
// - `::: torch_world.<path>` directives are replaced by the API reference that prepare.py renders into api.json;
// - `:material-<name>:` and `:octicons-arrow-right-24:` icons, with an optional `{ .class }`, become inline SVG;
// - links to `.ipynb` notebooks point at the pages prepare.py generates for them.
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import type MarkdownIt from 'markdown-it'

const require = createRequire(import.meta.url)
const API = path.resolve(__dirname, 'api.json')
// Kept in sync with DIRECTIVE in prepare.py.
const DIRECTIVE = /^::: (torch_world[\w.]*)[ \t]*\n((?:[ \t]+\S.*\n)*)/gm

let api: { mtime: number; pages: Record<string, string[]> } | undefined

function apiPages(): Record<string, string[]> {
  const mtime = fs.existsSync(API) ? fs.statSync(API).mtimeMs : 0
  if (!api || api.mtime !== mtime) {
    api = { mtime, pages: mtime ? JSON.parse(fs.readFileSync(API, 'utf8')) : {} }
  }
  return api.pages
}

const mdi: { icons: Record<string, { body: string; width?: number; height?: number }>; width?: number; height?: number } =
  require('@iconify-json/mdi/icons.json')

const OCTICONS: Record<string, string> = {
  'arrow-right-24':
    '<svg viewBox="0 0 24 24"><path fill="currentColor" d="M13.22 19.03a.75.75 0 0 1 0-1.06L18.19 13H3.75a.75.75 0 0 1 0-1.5h14.44l-4.97-4.97a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215l6.25 6.25a.75.75 0 0 1 0 1.06l-6.25 6.25a.75.75 0 0 1-1.06 0Z"/></svg>'
}

function icon(set: string, name: string): string | undefined {
  if (set === 'octicons') return OCTICONS[name]
  const data = mdi.icons[name]
  if (!data) return undefined
  const width = data.width ?? mdi.width ?? 24
  const height = data.height ?? mdi.height ?? 24
  return `<svg viewBox="0 0 ${width} ${height}">${data.body}</svg>`
}

const ICON = /^:(material|octicons)-([a-z0-9-]+):(?:\{([^}]*)\})?/

export function torchWorld(md: MarkdownIt) {
  md.core.ruler.after('normalize', 'tw_api', (state) => {
    // A language's copy of an API page, such as zh/api/nn.md, uses the English page's blocks.
    const path: string = state.env?.relativePath ?? ''
    const pages = apiPages()
    const blocks = pages[path] ?? pages[path.replace(/^[^/]+\//, '')]
    if (!blocks) return
    let i = 0
    state.src = state.src.replace(DIRECTIVE, (directive) => (i < blocks.length ? `${blocks[i++]}\n\n` : directive))
  })

  md.inline.ruler.before('emphasis', 'tw_icon', (state, silent) => {
    if (state.src.charCodeAt(state.pos) !== 0x3a /* : */) return false
    const match = ICON.exec(state.src.slice(state.pos))
    const svg = match && icon(match[1], match[2])
    if (!match || !svg) return false
    if (!silent) {
      const classes = (match[3] ?? '').match(/\.[\w-]+/g)?.map((c) => c.slice(1)) ?? []
      const token = state.push('html_inline', '', 0)
      token.content = `<span class="tw-icon ${classes.join(' ')}">${svg}</span>`
    }
    state.pos += match[0].length
    return true
  })

  md.core.ruler.push('tw_notebook_links', (state) => {
    for (const block of state.tokens) {
      for (const token of block.children ?? []) {
        const href = token.type === 'link_open' && token.attrGet('href')
        if (href && !/^[a-z]+:/i.test(href)) token.attrSet('href', href.replace(/\.ipynb(?=$|#)/, '.md'))
      }
    }
  })
}
