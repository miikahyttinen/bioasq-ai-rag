import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Source } from '../api/types'
import { linkCitations } from '../lib/citations'
import { usePassageDrawer } from '../lib/passageDrawer'
import { cx } from '../lib/format'

/** Markdown answer whose [n] citations become chips that open the cited source. */
export function Answer({
  text,
  sources,
  streaming = false,
}: {
  text: string
  sources: Source[]
  streaming?: boolean
}) {
  const openPassage = usePassageDrawer()
  const markdown = linkCitations(text, sources.length)

  return (
    <div className={cx('prose-answer', streaming && 'caret')}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const cite = href?.match(/^#cite-(\d+)$/)
            if (cite) {
              const n = Number(cite[1])
              const source = sources[n - 1]
              return (
                <button
                  type="button"
                  onClick={() =>
                    source && openPassage({ pmid: source.pmid, highlight: source.text, n })
                  }
                  title={source ? `PMID ${source.pmid}` : undefined}
                  className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-px items-center justify-center rounded-md border border-accent/40 bg-accent/10 px-1 align-baseline font-mono text-[0.7rem] font-semibold text-accent transition hover:bg-accent hover:text-on-accent"
                >
                  {n}
                </button>
              )
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="text-accent underline">
                {children}
              </a>
            )
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

/** Compact list of numbered sources under an answer. */
export function SourceList({
  sources,
  cited,
  inGold,
}: {
  sources: Source[]
  cited?: Set<number>
  inGold?: (source: Source) => boolean
}) {
  const openPassage = usePassageDrawer()
  return (
    <ol className="grid gap-2 sm:grid-cols-2">
      {sources.map((s) => {
        const gold = inGold?.(s)
        return (
          <li key={s.chunk_id}>
            <button
              onClick={() => openPassage({ pmid: s.pmid, highlight: s.text, n: s.n })}
              className={cx(
                'group flex h-full w-full gap-3 rounded-lg border bg-surface/60 p-3 text-left transition hover:border-accent',
                gold === true ? 'border-good/50' : 'border-line',
                cited && !cited.has(s.n) && 'opacity-60',
              )}
            >
              <span
                className={cx(
                  'flex h-6 w-6 shrink-0 items-center justify-center rounded-md font-mono text-xs font-semibold',
                  gold === true ? 'bg-good/15 text-good' : 'bg-accent/10 text-accent',
                )}
              >
                {s.n}
              </span>
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-x-2 text-xs text-muted">
                  <span className="font-mono">PMID {s.pmid}</span>
                  {gold === true && <span className="text-good">✓ gold</span>}
                  {gold === false && <span>not in gold set</span>}
                  {cited && !cited.has(s.n) && <span>· not cited</span>}
                </span>
                <span className="mt-1 line-clamp-2 block text-sm text-ink-2 group-hover:text-ink">
                  {s.text}
                </span>
                {s.cluster_label && (
                  <span className="mt-1 block truncate text-[0.7rem] text-muted">
                    {s.cluster_label}
                  </span>
                )}
              </span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}
