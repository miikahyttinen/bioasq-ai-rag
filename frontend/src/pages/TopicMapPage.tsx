import { useQuery } from '@tanstack/react-query'
import Plotly from 'plotly.js-dist-min'
import { useMemo, useState } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Cluster, TopicPoint } from '../api/types'
import { ChatIcon, SearchIcon } from '../components/icons'
import { usePassageDrawer } from '../lib/passageDrawer'
import { Button, ErrorNote, Spinner } from '../components/ui'
import { cx } from '../lib/format'
import { cssVar, useDocumentTheme, type Theme } from '../lib/theme'

const Plot = createPlotlyComponent(Plotly)

/** scattergl needs WebGL; fall back to (slower) SVG markers without it. */
const HAS_WEBGL = (() => {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') ?? canvas.getContext('webgl'))
  } catch {
    return false
  }
})()

/** Largest clusters get a direct label on the map; the rest rely on hover and the list. */
const LABELED_CLUSTERS = 12
/** Approximate label size in map units at the initial zoom (measured: ~0.27 units per character). */
const UNITS_PER_CHAR = 0.27
const LABEL_HEIGHT = 1.6

function placeLabels(bySize: Cluster[]): Cluster[] {
  const placed: Cluster[] = []
  const width = (c: Cluster) => c.label.length * UNITS_PER_CHAR
  for (const c of bySize) {
    if (placed.length >= LABELED_CLUSTERS || c.x == null || c.y == null) continue
    const clashes = placed.some(
      (p) =>
        Math.abs(p.x! - c.x!) < (width(p) + width(c)) / 2 &&
        Math.abs(p.y! - c.y!) < LABEL_HEIGHT,
    )
    if (!clashes) placed.push(c)
  }
  return placed
}

/** Initial view framing the central 99% of points, so a few stragglers don't shrink it. */
function centralRange(values: number[]): [number, number] {
  const sorted = [...values].sort((a, b) => a - b)
  const lo = sorted[Math.floor(sorted.length * 0.005)]
  const hi = sorted[Math.ceil(sorted.length * 0.995) - 1]
  const pad = (hi - lo) * 0.04
  return [lo - pad, hi + pad]
}

export default function TopicMapPage() {
  const points = useQuery({ queryKey: ['topic-map'], queryFn: api.topicMap })
  const clusters = useQuery({ queryKey: ['clusters'], queryFn: api.clusters })

  if (points.isLoading || clusters.isLoading) return <Spinner label="Loading 27,972 points" />
  if (points.error || clusters.error) return <ErrorNote error={points.error ?? clusters.error} />
  return <TopicMap points={points.data!} clusters={clusters.data!} />
}

function TopicMap({ points, clusters }: { points: TopicPoint[]; clusters: Cluster[] }) {
  const theme = useDocumentTheme()
  const navigate = useNavigate()
  const openPassage = usePassageDrawer()
  const [params, setParams] = useSearchParams()
  const selected = params.get('cluster') != null ? Number(params.get('cluster')) : null
  const [filter, setFilter] = useState('')
  const [showNoise, setShowNoise] = useState(true)

  const byId = useMemo(() => new Map(clusters.map((c) => [c.cluster_id, c])), [clusters])
  const topics = clusters.filter((c) => c.cluster_id >= 0)
  const noise = byId.get(-1)
  const selectedCluster = selected != null ? byId.get(selected) : undefined

  const select = (id: number | null) =>
    setParams(id == null || id === selected ? {} : { cluster: String(id) })

  const figure = useMemo(() => {
    const color = mapColors(theme)
    const groups = { noise: [] as TopicPoint[], base: [] as TopicPoint[], hit: [] as TopicPoint[] }
    for (const p of points) {
      if (selected != null && p.cluster_id === selected) groups.hit.push(p)
      else if (p.cluster_id === -1) groups.noise.push(p)
      else groups.base.push(p)
    }
    const dim = selected != null

    const trace = (
      name: string,
      pts: TopicPoint[],
      marker: Record<string, unknown>,
      visible = true,
    ) => ({
      type: HAS_WEBGL ? 'scattergl' : 'scatter',
      mode: 'markers',
      name,
      visible,
      x: pts.map((p) => p.x),
      y: pts.map((p) => p.y),
      customdata: pts.map((p) => {
        const own = byId.get(p.cluster_id)
        const near = byId.get(p.nearest_cluster)
        const label =
          p.cluster_id === -1 ? `Unclustered · near ${near?.label ?? '—'}` : (own?.label ?? '')
        return [p.pmid, label, p.cluster_id]
      }),
      hovertemplate: '<b>%{customdata[1]}</b><br>PMID %{customdata[0]}<extra></extra>',
      marker,
    })

    const data = [
      trace(
        'Unclustered',
        groups.noise,
        { size: 3, color: color.noise, opacity: dim ? 0.35 : 0.7 },
        showNoise,
      ),
      trace('Topics', groups.base, { size: 4, color: color.point, opacity: dim ? 0.3 : 0.9 }),
      trace(selectedCluster?.label ?? 'Selected', groups.hit, {
        size: 6,
        color: color.accent,
        opacity: 0.95,
        line: { width: 1, color: color.surface },
      }),
    ]

    // Direct labels for the largest topics, skipping any that would overlap a label
    // already placed (dense centre), plus the selected topic.
    const labeled = placeLabels(topics.filter((c) => c.cluster_id !== selected))
    const annotations = [
      ...labeled.map((c) => ({
        x: c.x,
        y: c.y,
        text: c.label,
        showarrow: false,
        font: { size: 10.5, color: dim ? color.border : color.ink2 },
      })),
      ...(selectedCluster
        ? [
            {
              x: selectedCluster.x,
              y: selectedCluster.y,
              text: `<b>${selectedCluster.label}</b>`,
              // Offset above the cluster with a pointer so the label never hides its points.
              showarrow: true,
              arrowcolor: color.accent,
              arrowwidth: 1.5,
              arrowhead: 0,
              ax: 0,
              ay: -48,
              font: { size: 13, color: color.ink },
              bgcolor: color.surface,
              bordercolor: color.accent,
              borderpad: 4,
            },
          ]
        : []),
    ]

    const axis = { visible: false, showgrid: false, zeroline: false }
    const xRange = centralRange(points.map((p) => p.x))
    const yRange = centralRange(points.map((p) => p.y))
    return {
      data,
      layout: {
        autosize: true,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { ...axis, range: xRange },
        yaxis: { ...axis, range: yRange, scaleanchor: 'x' },
        showlegend: false,
        hovermode: 'closest',
        hoverlabel: {
          bgcolor: color.surface,
          bordercolor: color.accent,
          font: { color: color.ink, family: 'Inter Variable, system-ui' },
        },
        annotations,
        dragmode: 'pan',
        uirevision: 'topic-map', // keep zoom/pan across re-renders
      },
    }
  }, [points, byId, topics, selected, selectedCluster, showNoise, theme])

  const visibleTopics = topics.filter((c) =>
    c.label.toLowerCase().includes(filter.trim().toLowerCase()),
  )

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight">Topic map</h1>
          <p className="mt-1 text-sm text-ink-2">
            {points.length.toLocaleString()} abstracts placed by meaning (UMAP) and grouped into{' '}
            {topics.length} topics (HDBSCAN). Scroll to zoom, drag to pan, click a point to read
            it.
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm text-ink-2">
          <input
            type="checkbox"
            checked={showNoise}
            onChange={(e) => setShowNoise(e.target.checked)}
            className="accent-[var(--accent)]"
          />
          Show unclustered ({noise?.size.toLocaleString()})
        </label>
      </header>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="panel relative h-[62vh] min-h-[420px] overflow-hidden p-1">
          <Plot
            data={figure.data}
            layout={figure.layout}
            config={{
              displaylogo: false,
              scrollZoom: true,
              responsive: true,
              modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toImage'],
            }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
            onClick={(event: { points?: { customdata: [number, string, number] }[] }) => {
              const point = event.points?.[0]
              if (point) openPassage({ pmid: point.customdata[0] })
            }}
          />
          <div className="pointer-events-none absolute bottom-3 left-3 flex gap-4 rounded-lg border border-line bg-panel px-3 py-2 text-xs text-ink-2 backdrop-blur">
            <LegendDot cls="bg-[var(--map-point)]" label="In a topic" />
            <LegendDot cls="bg-[var(--map-noise)]" label="Unclustered" />
            {selectedCluster && <LegendDot cls="bg-accent" label="Selected topic" />}
          </div>
        </div>

        <aside className="panel flex flex-col xl:max-h-[62vh]">
          {selectedCluster ? (
            <div className="border-b border-line p-4">
              <div className="label-caps">Selected topic</div>
              <h2 className="mt-1 font-display text-lg font-semibold">{selectedCluster.label}</h2>
              <p className="mt-1 text-sm text-ink-2">{selectedCluster.description}</p>
              <p className="mt-2 font-mono text-xs text-muted">
                {selectedCluster.size.toLocaleString()} abstracts
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  size="sm"
                  onClick={() =>
                    navigate('/', {
                      state: {
                        prompt: `Give me an overview of the research on ${selectedCluster.label.toLowerCase()}: the main findings, mechanisms and open questions.`,
                      },
                    })
                  }
                >
                  <ChatIcon size={14} /> Chat about this topic
                </Button>
                <Button size="sm" variant="ghost" onClick={() => select(null)}>
                  Clear
                </Button>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                <span className="label-caps w-full">Most typical abstracts</span>
                {selectedCluster.representative_pmids.slice(0, 6).map((pmid) => (
                  <button
                    key={pmid}
                    onClick={() => openPassage({ pmid })}
                    className="rounded-md border border-line px-2 py-0.5 font-mono text-xs text-ink-2 hover:border-accent hover:text-ink"
                  >
                    {pmid}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <p className="border-b border-line p-4 text-sm text-muted">
              Select a topic to highlight it on the map.
            </p>
          )}

          <div className="border-b border-line p-3">
            <div className="relative">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" />
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder={`Filter ${topics.length} topics…`}
                className="input pl-10"
                aria-label="Filter topics"
              />
            </div>
          </div>

          <ul className="flex-1 overflow-y-auto p-2">
            {visibleTopics.map((c) => (
              <li key={c.cluster_id}>
                <button
                  onClick={() => select(c.cluster_id)}
                  className={cx(
                    'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition',
                    c.cluster_id === selected
                      ? 'bg-raised text-ink glow'
                      : 'text-ink-2 hover:bg-raised/60 hover:text-ink',
                  )}
                >
                  <span className="min-w-0 flex-1 truncate">{c.label}</span>
                  <span className="relative h-1.5 w-16 shrink-0 overflow-hidden rounded-full bg-raised">
                    <span
                      className="absolute inset-y-0 left-0 rounded-full bg-accent/70"
                      style={{ width: `${(c.size / topics[0].size) * 100}%` }}
                    />
                  </span>
                  <span className="w-10 shrink-0 text-right font-mono text-xs text-muted">
                    {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  )
}

/** Plotly needs concrete colors; read the CSS tokens for the active theme. */
function mapColors(_theme: Theme) {
  return {
    point: cssVar('--map-point'),
    noise: cssVar('--map-noise'),
    accent: cssVar('--accent'),
    ink: cssVar('--ink'),
    ink2: cssVar('--ink-2'),
    surface: cssVar('--surface'),
    border: cssVar('--border-strong'),
  }
}

function LegendDot({ cls, label }: { cls: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={cx('h-2 w-2 rounded-full', cls)} aria-hidden="true" />
      {label}
    </span>
  )
}
