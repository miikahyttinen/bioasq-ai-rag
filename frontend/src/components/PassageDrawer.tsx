import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { PassageContext, type Opened } from '../lib/passageDrawer'
import { CloseIcon, ExternalIcon } from './icons'
import { Button, ErrorNote, Spinner } from './ui'


export function PassageProvider({ children }: { children: ReactNode }) {
  const [opened, setOpened] = useState<Opened | null>(null)
  const open = useCallback((o: Opened) => setOpened(o), [])

  return (
    <PassageContext.Provider value={open}>
      {children}
      {opened && <Drawer opened={opened} onClose={() => setOpened(null)} />}
    </PassageContext.Provider>
  )
}

function Drawer({ opened, onClose }: { opened: Opened; onClose: () => void }) {
  const navigate = useNavigate()
  const { data, error, isLoading } = useQuery({
    queryKey: ['passage', opened.pmid],
    queryFn: () => api.passage(opened.pmid),
  })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const topic =
    data && (data.cluster_id === -1 ? data.nearest_cluster_label : data.cluster_label)

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true">
      <button
        className="absolute inset-0 bg-black/50 backdrop-blur-[2px]"
        aria-label="Close passage"
        onClick={onClose}
      />
      <aside className="relative flex h-full w-full max-w-xl flex-col border-l border-line-strong bg-panel-solid shadow-2xl">
        <header className="flex items-start justify-between gap-3 border-b border-line p-5">
          <div>
            <div className="label-caps">
              {opened.n ? `Source [${opened.n}] · ` : ''}PubMed abstract
            </div>
            <div className="mt-1 font-display text-xl font-semibold">PMID {opened.pmid}</div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </Button>
        </header>

        <div className="flex-1 overflow-y-auto p-5">
          {isLoading && <Spinner label="Loading abstract" />}
          {error && <ErrorNote error={error} />}
          {data && (
            <>
              {topic && (
                <button
                  className="mb-4 inline-flex items-center gap-2 rounded-full border border-line-strong px-3 py-1 text-xs text-ink-2 hover:border-accent hover:text-ink"
                  onClick={() => {
                    const cluster =
                      data.cluster_id === -1 ? data.nearest_cluster_id : data.cluster_id
                    onClose()
                    navigate(`/topics?cluster=${cluster}`)
                  }}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden="true" />
                  {data.cluster_id === -1 ? `Near topic: ${topic}` : `Topic: ${topic}`}
                </button>
              )}
              <HighlightedText text={data.text} highlight={opened.highlight} />
            </>
          )}
        </div>

        <footer className="border-t border-line p-4">
          <a
            href={`https://pubmed.ncbi.nlm.nih.gov/${opened.pmid}/`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 text-sm text-accent hover:underline"
          >
            Open on PubMed <ExternalIcon size={14} />
          </a>
        </footer>
      </aside>
    </div>
  )
}

/** Shows the full abstract and marks the retrieved chunk when it is a part of it. */
function HighlightedText({ text, highlight }: { text: string; highlight?: string }) {
  const start = highlight && highlight !== text ? text.indexOf(highlight) : -1
  if (start < 0) {
    return <p className="whitespace-pre-line leading-relaxed text-ink">{text}</p>
  }
  const end = start + highlight!.length
  return (
    <p className="whitespace-pre-line leading-relaxed text-ink-2">
      {text.slice(0, start)}
      <mark className="rounded bg-accent/15 px-0.5 text-ink ring-1 ring-accent/30">
        {text.slice(start, end)}
      </mark>
      {text.slice(end)}
      <span className="mt-3 block text-xs text-muted">
        Highlighted: the retrieved chunk used as the source.
      </span>
    </p>
  )
}
