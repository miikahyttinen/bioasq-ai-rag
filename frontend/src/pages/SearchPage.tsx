import { useQueries } from '@tanstack/react-query'
import { useMemo, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Source, Strategy } from '../api/types'
import { STRATEGIES } from '../api/types'
import { SearchIcon } from '../components/icons'
import { usePassageDrawer } from '../lib/passageDrawer'
import { Button, ErrorNote, Segmented } from '../components/ui'
import { cx } from '../lib/format'
import { STRATEGY_META } from '../lib/strategies'

const EXAMPLES = [
  'BRCA1 mutations breast cancer risk',
  'statin-induced myopathy',
  'CRISPR off-target effects',
  'IL-6 in rheumatoid arthritis',
]

export default function SearchPage() {
  const [params, setParams] = useSearchParams()
  const query = params.get('q') ?? ''
  const k = Number(params.get('k') ?? 10)
  const [draft, setDraft] = useState(query)

  const results = useQueries({
    queries: STRATEGIES.map((strategy) => ({
      queryKey: ['search', query, strategy, k],
      queryFn: () => api.search(query, strategy, k),
      enabled: query.length > 0,
    })),
  })

  // pmid -> strategies that returned it, for the overlap markers.
  const foundBy = useMemo(() => {
    const map = new Map<number, Set<Strategy>>()
    results.forEach((r, i) =>
      r.data?.results.forEach((s) => {
        if (!map.has(s.pmid)) map.set(s.pmid, new Set())
        map.get(s.pmid)!.add(STRATEGIES[i])
      }),
    )
    return map
  }, [results])

  const inAll = [...foundBy.values()].filter((set) => set.size === 3).length

  function submit(e?: FormEvent, value = draft) {
    e?.preventDefault()
    if (value.trim()) setParams({ q: value.trim(), k: String(k) })
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-2xl font-semibold tracking-tight">Compare retrieval</h1>
        <p className="mt-1 text-sm text-ink-2">
          One query, three search strategies side by side. No LLM involved: this is what the
          model would get as sources.
        </p>
      </header>

      <form onSubmit={submit} className="panel flex flex-col gap-3 p-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" />
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Search 27,972 PubMed abstracts…"
            className="input pl-10"
            aria-label="Search query"
          />
        </div>
        <Segmented
          label="Results per strategy"
          value={k}
          onChange={(value) => query && setParams({ q: query, k: String(value) })}
          options={[5, 10, 20].map((n) => ({ value: n, label: `Top ${n}` }))}
        />
        <Button type="submit" disabled={!draft.trim()}>
          Search
        </Button>
      </form>

      {!query && (
        <div className="flex flex-wrap gap-2">
          <span className="label-caps self-center">Try</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => {
                setDraft(ex)
                submit(undefined, ex)
              }}
              className="rounded-full border border-line px-3 py-1 text-sm text-ink-2 hover:border-accent hover:text-ink"
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      {query && foundBy.size > 0 && (
        <p className="text-sm text-ink-2">
          <span className="font-mono text-ink">{foundBy.size}</span> distinct articles across the
          three lists · <span className="font-mono text-ink">{inAll}</span> found by all three.
          Markers on each result show which other strategies also found it.
        </p>
      )}

      {query && (
        <div className="grid gap-4 lg:grid-cols-3">
          {STRATEGIES.map((strategy, i) => (
            <Column
              key={strategy}
              strategy={strategy}
              loading={results[i].isLoading}
              error={results[i].error}
              sources={results[i].data?.results}
              seconds={results[i].data?.seconds}
              foundBy={foundBy}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function Column({
  strategy,
  loading,
  error,
  sources,
  seconds,
  foundBy,
}: {
  strategy: Strategy
  loading: boolean
  error: unknown
  sources?: Source[]
  seconds?: number
  foundBy: Map<number, Set<Strategy>>
}) {
  const meta = STRATEGY_META[strategy]
  const openPassage = usePassageDrawer()

  return (
    <section className="panel flex flex-col overflow-hidden">
      <header className="flex items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <span
            className={cx('h-2.5 w-2.5 rounded-full', meta.dot)}
            style={{ boxShadow: `0 0 10px var(${meta.cssVar})` }}
            aria-hidden="true"
          />
          <h2 className="font-display font-semibold">{meta.label}</h2>
        </div>
        {seconds != null && <span className="font-mono text-xs text-muted">{seconds}s</span>}
      </header>
      <p className="border-b border-line px-4 py-2 text-xs text-muted">{meta.blurb}</p>

      <div className="flex-1 space-y-2 p-3">
        {loading && <div className="scanbar h-24 rounded-lg bg-raised/50" />}
        {error != null && <ErrorNote error={error} />}
        {sources?.length === 0 && <p className="p-2 text-sm text-muted">No matches.</p>}
        {sources?.map((s) => {
          const others = STRATEGIES.filter((o) => o !== strategy && foundBy.get(s.pmid)?.has(o))
          return (
            <button
              key={s.chunk_id}
              onClick={() => openPassage({ pmid: s.pmid, highlight: s.text })}
              className="group block w-full rounded-lg border border-line bg-surface/60 p-3 text-left transition hover:border-accent"
            >
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="flex items-center gap-2">
                  <span className="font-mono font-semibold text-ink">#{s.n}</span>
                  <span className="font-mono text-muted">PMID {s.pmid}</span>
                </span>
                <span className="flex items-center gap-1" title="Also found by">
                  {others.map((o) => (
                    <span
                      key={o}
                      className="inline-flex items-center gap-1 rounded border border-line px-1 font-mono text-[0.65rem] text-ink-2"
                    >
                      <span className={cx('h-1.5 w-1.5 rounded-full', STRATEGY_META[o].dot)} />
                      {STRATEGY_META[o].short}
                    </span>
                  ))}
                  <span className="ml-1 font-mono text-muted">{s.score.toFixed(3)}</span>
                </span>
              </div>
              <p className="mt-2 line-clamp-3 text-sm text-ink-2 group-hover:text-ink">{s.text}</p>
              {s.cluster_label && (
                <p className="mt-1.5 truncate text-[0.7rem] text-muted">{s.cluster_label}</p>
              )}
            </button>
          )
        })}
      </div>
    </section>
  )
}
