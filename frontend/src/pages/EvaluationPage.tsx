import { useQuery } from '@tanstack/react-query'
import { useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import type { EvalResults, Strategy } from '../api/types'
import { STRATEGIES } from '../api/types'
import { TableIcon } from '../components/icons'
import { ErrorNote, Panel, ScoreBadge, Spinner, StatTile } from '../components/ui'
import { cx, pct } from '../lib/format'
import { STRATEGY_META } from '../lib/strategies'
import { cssVar, useDocumentTheme } from '../lib/theme'

type Retrieval = NonNullable<EvalResults['retrieval']>
type Answers = NonNullable<EvalResults['answers']>

export default function EvaluationPage() {
  const { data, error, isLoading } = useQuery({
    queryKey: ['eval'],
    queryFn: () => api.evalResults(),
  })
  if (isLoading) return <Spinner label="Loading evaluation results" />
  if (error) return <ErrorNote error={error} />
  if (!data?.retrieval) {
    return <ErrorNote error="No evaluation results yet. Run: uv run python -m scripts.06_evaluate" />
  }
  return <Dashboard data={data} retrieval={data.retrieval} />
}

/** Chart colors resolved from CSS tokens (re-read when the theme changes). */
function useChartColors() {
  const theme = useDocumentTheme()
  return useMemo(
    () => ({
      dense: cssVar('--series-dense'),
      bm25: cssVar('--series-bm25'),
      hybrid: cssVar('--series-hybrid'),
      rag: cssVar('--series-rag'),
      norag: cssVar('--series-norag'),
      grid: cssVar('--gridline'),
      axis: cssVar('--axis'),
      muted: cssVar('--muted'),
      ink2: cssVar('--ink-2'),
      surface: cssVar('--surface'),
      theme,
    }),
    [theme],
  )
}

function Dashboard({ data, retrieval }: { data: EvalResults; retrieval: Retrieval }) {
  const best = retrieval.best_strategy
  const b = retrieval.strategies[best]
  const answers = data.answers

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-2xl font-semibold tracking-tight">Evaluation</h1>
        <p className="mt-1 text-sm text-ink-2">
          Retrieval measured on {retrieval.n_questions.toLocaleString()} questions against the
          expert gold sets; answer quality judged on a fixed sample of{' '}
          {answers?.n_questions ?? '—'}.
        </p>
      </header>

      {answers && <Hero answers={answers} />}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Best retriever"
          value={STRATEGY_META[best].label}
          hint={STRATEGY_META[best].blurb}
          accent
        />
        <StatTile label="Recall@10" value={pct(b['recall@10'], 1)} hint="share of gold articles in the top 10" />
        <StatTile label="Hit-rate@10" value={pct(b['hit@10'], 1)} hint="≥ 1 gold article in the top 10" />
        <StatTile label="MRR" value={b.mrr.toFixed(3)} hint="1 / rank of the first gold article" />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <RecallChart retrieval={retrieval} />
        <RankChart retrieval={retrieval} />
      </div>

      {answers && <CorrectnessChart answers={answers} />}

      <div className="grid gap-6 xl:grid-cols-2">
        <WorstRetrieval data={data} />
        <RagDifferences data={data} />
      </div>

      {answers && (
        <p className="text-xs text-muted">
          Judge: {answers.config.model}, same model as the answers ({answers.note}). Groundedness
          averaged {answers.rag.groundedness ?? '—'}/5 and does not discriminate here. BioASQ gold
          sets are not exhaustive, so a "miss" can still be a relevant paper.
        </p>
      )}
    </div>
  )
}

function Hero({ answers }: { answers: Answers }) {
  return (
    <div className="panel glow relative overflow-hidden p-6 sm:p-8">
      <div className="label-caps">What retrieval adds</div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <span className="font-display text-5xl font-semibold tracking-tight text-gradient sm:text-6xl">
          {pct(answers.rag.share_correct_4plus)}
        </span>
        <span className="text-lg text-ink-2">
          of answers correct with RAG, vs{' '}
          <span className="font-semibold text-ink">{pct(answers.no_rag.share_correct_4plus)}</span>{' '}
          from the model alone
        </span>
      </div>
      <p className="mt-2 text-sm text-muted">
        Judge score ≥ 4 of 5 · mean correctness {answers.rag.correctness} vs{' '}
        {answers.no_rag.correctness} · {answers.n_questions} questions · {answers.config.strategy}{' '}
        retrieval, k = {answers.config.k}
      </p>
    </div>
  )
}

function ChartPanel({
  title,
  subtitle,
  table,
  children,
}: {
  title: string
  subtitle: string
  table: ReactNode
  children: ReactNode
}) {
  const [showTable, setShowTable] = useState(false)
  return (
    <Panel
      title={title}
      actions={
        <button
          onClick={() => setShowTable((s) => !s)}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted hover:bg-raised hover:text-ink"
          aria-pressed={showTable}
        >
          <TableIcon size={14} /> {showTable ? 'Chart' : 'Table'}
        </button>
      }
    >
      <p className="-mt-2 mb-4 text-sm text-ink-2">{subtitle}</p>
      {showTable ? table : children}
    </Panel>
  )
}

function ChartTooltip({
  active,
  payload,
  label,
  format,
}: {
  active?: boolean
  payload?: { name: string; value: number; color: string }[]
  label?: string | number
  format: (v: number) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-line-strong bg-panel-solid px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-medium text-ink">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-ink-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="flex-1">{p.name}</span>
          <span className="font-mono text-ink">{format(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

function SeriesLegend({ items }: { items: { label: string; color: string; line?: boolean }[] }) {
  return (
    <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5">
          <span
            className={item.line ? 'h-0.5 w-3.5 rounded-full' : 'h-2.5 w-2.5 rounded-full'}
            style={{ background: item.color }}
            aria-hidden="true"
          />
          {item.label}
        </span>
      ))}
    </div>
  )
}

function DataTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-muted">
            {head.map((h, i) => (
              <th key={h} className={cx('pb-2 font-normal', i > 0 && 'text-right')}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {rows.map((row) => (
            <tr key={String(row[0])} className="border-t border-line">
              {row.map((cell, i) => (
                <td key={i} className={cx('py-2', i > 0 ? 'text-right font-mono text-ink' : 'text-ink-2')}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const KS = [1, 5, 10, 20] as const

function RecallChart({ retrieval }: { retrieval: Retrieval }) {
  const c = useChartColors()
  const rows = KS.map((k) => ({
    k: `@${k}`,
    ...Object.fromEntries(
      STRATEGIES.map((s) => [s, retrieval.strategies[s][`recall@${k}` as 'recall@1']]),
    ),
  }))
  return (
    <ChartPanel
      title="Recall@k"
      subtitle="Share of each question's gold articles found in the top k, averaged."
      table={
        <DataTable
          head={['Strategy', ...KS.map((k) => `Recall@${k}`)]}
          rows={STRATEGIES.map((s) => [
            STRATEGY_META[s].label,
            ...KS.map((k) => retrieval.strategies[s][`recall@${k}` as 'recall@1'].toFixed(3)),
          ])}
        />
      }
    >
      <div className="h-64">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 44, bottom: 0, left: -12 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="k" stroke={c.axis} tick={{ fill: c.muted, fontSize: 12 }} tickLine={false} />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v) => `${v * 100}%`}
            stroke={c.axis}
            tick={{ fill: c.muted, fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={<ChartTooltip format={(v) => pct(v, 1)} />}
            cursor={{ stroke: c.axis }}
          />
          {STRATEGIES.map((s) => (
            <Line
              key={s}
              dataKey={s}
              name={STRATEGY_META[s].label}
              stroke={c[s]}
              strokeWidth={2}
              dot={{ r: 4, fill: c[s], stroke: c.surface, strokeWidth: 2 }}
              activeDot={{ r: 6, stroke: c.surface, strokeWidth: 2 }}
              isAnimationActive={false}
            >
              <LabelList
                dataKey={s}
                content={({ x, y, index, value }) =>
                  s === retrieval.best_strategy && index === KS.length - 1 ? (
                    <text x={Number(x) + 10} y={Number(y) + 4} fill={c.ink2} fontSize={11}>
                      {pct(Number(value))}
                    </text>
                  ) : null
                }
              />
            </Line>
          ))}
        </LineChart>
      </ResponsiveContainer>
      </div>
      <SeriesLegend items={STRATEGIES.map((s) => ({ label: STRATEGY_META[s].label, color: c[s], line: true }))} />
    </ChartPanel>
  )
}

function RankChart({ retrieval }: { retrieval: Retrieval }) {
  const c = useChartColors()
  const metrics = [
    { key: 'mrr', label: 'MRR' },
    { key: 'hit@10', label: 'Hit-rate@10' },
  ] as const
  const rows = metrics.map((m) => ({
    metric: m.label,
    ...Object.fromEntries(STRATEGIES.map((s) => [s, retrieval.strategies[s][m.key]])),
  }))
  return (
    <ChartPanel
      title="Ranking quality"
      subtitle="How early the first gold article appears (MRR), and how often one is in the top 10."
      table={
        <DataTable
          head={['Strategy', 'MRR', 'Hit-rate@10']}
          rows={STRATEGIES.map((s) => [
            STRATEGY_META[s].label,
            retrieval.strategies[s].mrr.toFixed(3),
            retrieval.strategies[s]['hit@10'].toFixed(3),
          ])}
        />
      }
    >
      <div className="h-64">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 20, right: 8, bottom: 0, left: -12 }} barGap={2}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="metric" stroke={c.axis} tick={{ fill: c.muted, fontSize: 12 }} tickLine={false} />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            stroke={c.axis}
            tick={{ fill: c.muted, fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={<ChartTooltip format={(v) => v.toFixed(3)} />}
            cursor={{ fill: c.grid, opacity: 0.4 }}
          />
          {STRATEGIES.map((s: Strategy) => (
            <Bar
              key={s}
              dataKey={s}
              name={STRATEGY_META[s].label}
              fill={c[s]}
              maxBarSize={24}
              radius={[4, 4, 0, 0]}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      </div>
      <SeriesLegend items={STRATEGIES.map((s) => ({ label: STRATEGY_META[s].label, color: c[s] }))} />
    </ChartPanel>
  )
}

function CorrectnessChart({ answers }: { answers: Answers }) {
  const c = useChartColors()
  const scores = ['1', '2', '3', '4', '5']
  const rows = scores.map((s) => ({
    score: `${s} / 5`,
    rag: answers.rag.correctness_distribution[s] ?? 0,
    norag: answers.no_rag.correctness_distribution[s] ?? 0,
  }))
  return (
    <ChartPanel
      title="Answer correctness: RAG vs model alone"
      subtitle={`Number of answers per judge score (${answers.n_questions} questions). RAG moves answers from wrong (1–2) to fully correct (5).`}
      table={
        <DataTable
          head={['Score', 'With RAG', 'Model alone']}
          rows={rows.map((r) => [r.score, r.rag, r.norag])}
        />
      }
    >
      <div className="h-64">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 20, right: 8, bottom: 0, left: -12 }} barGap={2}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="score" stroke={c.axis} tick={{ fill: c.muted, fontSize: 12 }} tickLine={false} />
          <YAxis stroke={c.axis} tick={{ fill: c.muted, fontSize: 12 }} tickLine={false} axisLine={false} allowDecimals={false} />
          <Tooltip
            content={<ChartTooltip format={(v) => `${v} answers`} />}
            cursor={{ fill: c.grid, opacity: 0.4 }}
          />
          {(
            [
              ['rag', 'With RAG'],
              ['norag', 'Model alone'],
            ] as const
          ).map(([key, name]) => (
            <Bar key={key} dataKey={key} name={name} fill={c[key]} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
              <LabelList
                dataKey={key}
                position="top"
                fill={c.ink2}
                fontSize={11}
                formatter={(v: unknown) => (Number(v) >= 10 ? String(v) : '')}
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      </div>
      <SeriesLegend
        items={[
          { label: 'With RAG', color: c.rag },
          { label: 'Model alone', color: c.norag },
        ]}
      />
    </ChartPanel>
  )
}

function WorstRetrieval({ data }: { data: EvalResults }) {
  const { strategy, questions } = data.worst_retrieval
  return (
    <Panel title={`Hardest questions for ${strategy ? STRATEGY_META[strategy].label : '—'} retrieval`}>
      <p className="-mt-2 mb-3 text-sm text-ink-2">
        Lowest recall@10: none of the expert-marked articles were found. Click to study why.
      </p>
      <ul className="divide-y divide-line">
        {questions.slice(0, 12).map((q) => (
          <li key={q.qid}>
            <Link
              to={`/questions?qid=${q.qid}`}
              className="flex items-center gap-3 py-2 text-sm text-ink-2 hover:text-ink"
            >
              <span className="min-w-0 flex-1 truncate">{q.question}</span>
              <span className="shrink-0 font-mono text-xs text-muted">
                R@20 {q['recall@20'].toFixed(2)} · {q.n_gold} gold
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </Panel>
  )
}

function RagDifferences({ data }: { data: EvalResults }) {
  const [mode, setMode] = useState<'helped' | 'hurt'>('helped')
  const records = data.answer_records
    .filter((r) => r.rag.correctness != null && r.no_rag.correctness != null)
    .map((r) => ({ ...r, delta: r.rag.correctness! - r.no_rag.correctness! }))
  const shown =
    mode === 'helped'
      ? records.filter((r) => r.delta > 0).sort((a, b) => b.delta - a.delta)
      : records.filter((r) => r.delta < 0).sort((a, b) => a.delta - b.delta)
  const helped = records.filter((r) => r.delta > 0).length
  const hurt = records.filter((r) => r.delta < 0).length

  return (
    <Panel
      title="Where retrieval changed the answer"
      actions={
        <div className="flex gap-1 text-xs">
          {(
            [
              ['helped', `Helped (${helped})`],
              ['hurt', `Hurt (${hurt})`],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              onClick={() => setMode(value)}
              className={cx(
                'rounded-md px-2 py-1',
                mode === value ? 'bg-raised text-ink glow' : 'text-muted hover:text-ink',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      }
    >
      <ul className="divide-y divide-line">
        {shown.slice(0, 12).map((r) => (
          <li key={r.qid}>
            <Link
              to={`/questions?qid=${r.qid}`}
              className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 text-sm text-ink-2 hover:text-ink"
            >
              <span className="min-w-0 flex-1 basis-60 truncate">{r.question}</span>
              <span className="flex shrink-0 gap-1.5">
                <ScoreBadge label="RAG" score={r.rag.correctness} />
                <ScoreBadge label="alone" score={r.no_rag.correctness} />
              </span>
            </Link>
          </li>
        ))}
        {shown.length === 0 && <li className="py-2 text-sm text-muted">None.</li>}
      </ul>
    </Panel>
  )
}
