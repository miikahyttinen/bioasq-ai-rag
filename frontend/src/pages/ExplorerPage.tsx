import { keepPreviousData, useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { QuestionDetail, Strategy } from '../api/types'
import { STRATEGIES } from '../api/types'
import { Answer, SourceList } from '../components/Answer'
import { CheckIcon, PlayIcon, SearchIcon } from '../components/icons'
import { usePassageDrawer } from '../lib/passageDrawer'
import { Button, ErrorNote, Panel, ScoreBadge, Segmented, Spinner, StrategyTag } from '../components/ui'
import { cx, pct } from '../lib/format'
import { STRATEGY_META } from '../lib/strategies'

const PAGE_SIZE = 30

export default function ExplorerPage() {
  const [params, setParams] = useSearchParams()
  const selected = params.get('qid') ? Number(params.get('qid')) : null
  const [search, setSearch] = useState('')
  const [debounced, setDebounced] = useState('')
  const [page, setPage] = useState(1)
  const [usableOnly, setUsableOnly] = useState(true)

  useEffect(() => {
    const t = setTimeout(() => {
      setDebounced(search)
      setPage(1)
    }, 250)
    return () => clearTimeout(t)
  }, [search])

  const list = useQuery({
    queryKey: ['questions', debounced, page, usableOnly],
    queryFn: () => api.questions(debounced, page, PAGE_SIZE, usableOnly),
    placeholderData: keepPreviousData,
  })
  const pages = list.data ? Math.max(1, Math.ceil(list.data.total / PAGE_SIZE)) : 1

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-2xl font-semibold tracking-tight">Question explorer</h1>
        <p className="mt-1 text-sm text-ink-2">
          Browse the 4,719 BioASQ questions, compare expert answers with the model's, and see
          which retrieved sources experts marked as relevant.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <aside className="panel flex flex-col lg:sticky lg:top-6 lg:max-h-[calc(100vh-8rem)]">
          <div className="space-y-3 border-b border-line p-3">
            <div className="relative">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter questions…"
                className="input pl-10"
                aria-label="Filter questions"
              />
            </div>
            <div className="flex items-center justify-between text-xs text-muted">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={usableOnly}
                  onChange={(e) => {
                    setUsableOnly(e.target.checked)
                    setPage(1)
                  }}
                  className="accent-[var(--accent)]"
                />
                Only with gold passages
              </label>
              <span className="font-mono">{list.data?.total.toLocaleString() ?? '…'}</span>
            </div>
          </div>

          <ul className="flex-1 overflow-y-auto p-2">
            {list.error && <ErrorNote error={list.error} />}
            {list.data?.items.map((q) => (
              <li key={q.qid}>
                <button
                  onClick={() => setParams({ qid: String(q.qid) })}
                  className={cx(
                    'w-full rounded-lg px-3 py-2.5 text-left text-sm transition',
                    q.qid === selected
                      ? 'bg-raised text-ink glow'
                      : 'text-ink-2 hover:bg-raised/60 hover:text-ink',
                  )}
                >
                  <span className="line-clamp-2">{q.question}</span>
                  <span className="mt-1 block font-mono text-[0.7rem] text-muted">
                    #{q.qid} · {q.n_gold} gold
                  </span>
                </button>
              </li>
            ))}
          </ul>

          <div className="flex items-center justify-between border-t border-line p-2 text-xs">
            <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              ← Prev
            </Button>
            <span className="font-mono text-muted">
              {page} / {pages}
            </span>
            <Button
              variant="ghost"
              size="sm"
              disabled={page >= pages}
              onClick={() => setPage(page + 1)}
            >
              Next →
            </Button>
          </div>
        </aside>

        <div className="min-w-0">
          {selected == null ? (
            <div className="panel flex h-64 items-center justify-center text-sm text-muted">
              Select a question to inspect it.
            </div>
          ) : (
            <QuestionView key={selected} qid={selected} />
          )}
        </div>
      </div>
    </div>
  )
}

function QuestionView({ qid }: { qid: number }) {
  const [strategy, setStrategy] = useState<Strategy>('hybrid')
  const detail = useQuery({ queryKey: ['question', qid], queryFn: () => api.question(qid) })
  const run = useMutation({ mutationFn: () => api.runQuestion(qid, strategy, 8) })

  if (detail.isLoading) return <Spinner label="Loading question" />
  if (detail.error) return <ErrorNote error={detail.error} />
  const q = detail.data!
  const gold = new Set(q.usable_relevant_ids)

  return (
    <div className="space-y-5">
      <Panel>
        <div className="label-caps">Question #{q.qid}</div>
        <h2 className="mt-2 font-display text-xl font-semibold leading-snug">{q.question}</h2>
        <div className="mt-4 rounded-lg border border-good/30 bg-good/5 p-4">
          <div className="label-caps mb-1 flex items-center gap-1.5 !text-good">
            <CheckIcon size={14} /> Expert answer
          </div>
          <p className="text-ink">{q.answer}</p>
        </div>
        <GoldList ids={q.usable_relevant_ids} missing={q.relevant_ids.length - gold.size} />
      </Panel>

      {q.retrieval && <StoredRetrieval detail={q} />}

      <Panel
        title="Run with RAG"
        actions={
          <div className="flex items-center gap-2">
            <Segmented
              label="Retrieval strategy"
              value={strategy}
              onChange={setStrategy}
              options={STRATEGIES.map((s) => ({ value: s, label: STRATEGY_META[s].label }))}
            />
            <Button onClick={() => run.mutate()} disabled={run.isPending}>
              <PlayIcon size={14} /> {run.isPending ? 'Running…' : 'Run'}
            </Button>
          </div>
        }
      >
        {!run.data && !run.isPending && !run.error && (
          <p className="text-sm text-muted">
            Retrieves 8 sources and generates an answer with the local model (~5–20 s). Sources
            in the gold set are marked green.
          </p>
        )}
        {run.isPending && <div className="scanbar rounded-md py-2 text-sm text-muted">Retrieving and generating…</div>}
        {run.error && <ErrorNote error={run.error} />}
        {run.data && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3 font-mono text-xs text-muted">
              <StrategyTag strategy={run.data.strategy} />
              <span>
                {run.data.hits}/{run.data.k} sources in gold set
              </span>
              <span>recall {pct(run.data.recall)}</span>
              <span>{run.data.seconds}s</span>
              <span>{run.data.model}</span>
            </div>
            <Answer text={run.data.answer} sources={run.data.sources} />
            <SourceList sources={run.data.sources} inGold={(s) => gold.has(s.pmid)} />
          </div>
        )}
      </Panel>

      {q.evaluation && <StoredEvaluation detail={q} />}
    </div>
  )
}

function GoldList({ ids, missing }: { ids: number[]; missing: number }) {
  const openPassage = usePassageDrawer()
  return (
    <div className="mt-4">
      <div className="label-caps mb-2">
        Gold passages ({ids.length}
        {missing > 0 ? `, ${missing} without abstract` : ''})
      </div>
      <div className="flex flex-wrap gap-1.5">
        {ids.map((pmid) => (
          <button
            key={pmid}
            onClick={() => openPassage({ pmid })}
            className="rounded-md border border-line px-2 py-0.5 font-mono text-xs text-ink-2 hover:border-good hover:text-ink"
          >
            {pmid}
          </button>
        ))}
      </div>
    </div>
  )
}

function StoredRetrieval({ detail }: { detail: QuestionDetail }) {
  const rows = STRATEGIES.map((s) => ({ strategy: s, m: detail.retrieval![s] }))
  return (
    <Panel title="Stored retrieval metrics (top 20)">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted">
              <th className="pb-2 font-normal">Strategy</th>
              {['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'MRR'].map((h) => (
                <th key={h} className="pb-2 text-right font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map(({ strategy, m }) => (
              <tr key={strategy} className="border-t border-line">
                <td className="py-2 font-sans">
                  <StrategyTag strategy={strategy} />
                </td>
                {[m['recall@1'], m['recall@5'], m['recall@10'], m['recall@20'], m.mrr].map(
                  (v, i) => (
                    <td key={i} className="py-2 text-right text-ink">
                      {v.toFixed(2)}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function StoredEvaluation({ detail }: { detail: QuestionDetail }) {
  const e = detail.evaluation!
  return (
    <Panel title="From the answer evaluation">
      <div className="grid gap-4 md:grid-cols-2">
        {(
          [
            ['With RAG', e.rag, true],
            ['Model alone', e.no_rag, false],
          ] as const
        ).map(([title, side, rag]) => (
          <div key={title} className="rounded-lg border border-line bg-surface/50 p-4">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="font-display font-semibold">{title}</span>
              <span className="flex gap-1.5">
                <ScoreBadge label="correct" score={side.judge.correctness} />
                {rag && <ScoreBadge label="grounded" score={side.judge.groundedness} />}
              </span>
            </div>
            <p className="line-clamp-[12] text-sm whitespace-pre-line text-ink-2">{side.answer}</p>
            {side.judge.reason && (
              <p className="mt-3 border-t border-line pt-2 text-xs text-muted">
                Judge: {side.judge.reason}
              </p>
            )}
          </div>
        ))}
      </div>
    </Panel>
  )
}
