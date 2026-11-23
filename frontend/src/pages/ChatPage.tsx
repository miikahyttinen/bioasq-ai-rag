import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { streamChat } from '../api/client'
import type { ChatDone, Source, Strategy } from '../api/types'
import { STRATEGIES } from '../api/types'
import { Answer, SourceList } from '../components/Answer'
import { SendIcon, SettingsIcon, SparkIcon, StopIcon } from '../components/icons'
import { Button, Segmented, StrategyTag } from '../components/ui'
import { citedNumbers } from '../lib/citations'
import { MODELS, STRATEGY_META } from '../lib/strategies'

function initialPrompt(location: { state: unknown; search: string }): string | null {
  return (
    (location.state as { prompt?: string } | null)?.prompt ??
    new URLSearchParams(location.search).get('q')
  )
}

interface Turn {
  id: number
  question: string
  answer: string
  sources: Source[]
  stage: 'retrieving' | 'generating' | 'done' | 'error' | 'stopped'
  done?: ChatDone
  error?: string
}

const EXAMPLES = [
  'What causes Pompe disease and how is it treated?',
  'How do PCSK9 inhibitors lower LDL cholesterol?',
  'Which genes are associated with Hirschsprung disease?',
  'What is the mechanism of action of CGRP-targeted migraine therapies?',
]

export default function ChatPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState(() => initialPrompt(location) ?? '')
  const [strategy, setStrategy] = useState<Strategy>('hybrid')
  const [k, setK] = useState(8)
  const [model, setModel] = useState(MODELS[0].id)
  const [think, setThink] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const busy = turns.some((t) => t.stage === 'retrieving' || t.stage === 'generating')

  // A prompt passed in ("Chat about this topic" or ?q=) prefills the input once;
  // then clear it from the history entry so a reload doesn't prefill again.
  useEffect(() => {
    if (initialPrompt(location)) {
      inputRef.current?.focus()
      navigate(location.pathname, { replace: true, state: null })
    }
  }, [location, navigate])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns])

  const update = (id: number, patch: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
    setTurns((all) =>
      all.map((t) => (t.id === id ? { ...t, ...(typeof patch === 'function' ? patch(t) : patch) } : t)),
    )

  async function send(question: string) {
    const text = question.trim()
    if (!text || busy) return
    const id = Date.now()
    const history = turns
      .filter((t) => t.stage === 'done')
      .flatMap((t) => [
        { role: 'user' as const, content: t.question },
        { role: 'assistant' as const, content: t.answer },
      ])
    setTurns((all) => [...all, { id, question: text, answer: '', sources: [], stage: 'retrieving' }])
    setInput('')

    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat(
        { message: text, history, strategy, k, model, think },
        {
          onSources: (sources) => update(id, { sources, stage: 'generating' }),
          onToken: (piece) => update(id, (t) => ({ answer: t.answer + piece })),
          onDone: (done) => update(id, { done, sources: done.sources, stage: 'done' }),
          onError: (message) => update(id, { error: message, stage: 'error' }),
        },
        controller.signal,
      )
    } catch (err) {
      if (controller.signal.aborted) update(id, { stage: 'stopped' })
      else update(id, { error: err instanceof Error ? err.message : String(err), stage: 'error' })
    } finally {
      abortRef.current = null
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    void send(input)
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send(input)
    }
  }

  return (
    <div className="mx-auto flex min-h-[calc(100vh-7rem)] max-w-4xl flex-col">
      <div className="flex-1 space-y-8 pb-6">
        {turns.length === 0 ? (
          <EmptyState onPick={(q) => void send(q)} />
        ) : (
          turns.map((turn) => <TurnView key={turn.id} turn={turn} />)
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={onSubmit} className="sticky bottom-4 z-10">
        {showSettings && (
          <div className="panel mb-2 grid gap-4 p-4 sm:grid-cols-2">
            <Setting label="Retrieval">
              <Segmented
                label="Retrieval strategy"
                value={strategy}
                onChange={setStrategy}
                options={STRATEGIES.map((s) => ({ value: s, label: STRATEGY_META[s].label }))}
              />
              <p className="mt-1 text-xs text-muted">{STRATEGY_META[strategy].blurb}</p>
            </Setting>
            <Setting label={`Sources (k = ${k})`}>
              <input
                type="range"
                min={1}
                max={8}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full accent-[var(--accent)]"
                aria-label="Number of sources"
              />
            </Setting>
            <Setting label="Model">
              <Segmented
                label="Model"
                value={model}
                onChange={setModel}
                options={MODELS.map((m) => ({ value: m.id, label: m.label }))}
              />
              <p className="mt-1 text-xs text-muted">
                {MODELS.find((m) => m.id === model)?.note}
              </p>
            </Setting>
            <Setting label="Reasoning">
              <label className="flex items-center gap-2 text-sm text-ink-2">
                <input
                  type="checkbox"
                  checked={think}
                  onChange={(e) => setThink(e.target.checked)}
                  className="accent-[var(--accent)]"
                />
                Think before answering (slower)
              </label>
            </Setting>
          </div>
        )}
        <div className="panel glow flex items-end gap-2 p-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowSettings((s) => !s)}
            aria-expanded={showSettings}
            aria-label="Chat settings"
            className="mb-0.5"
          >
            <SettingsIcon />
          </Button>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="Ask a biomedical question…"
            className="max-h-40 min-h-[2.5rem] flex-1 resize-none bg-transparent px-2 py-2 text-ink placeholder:text-muted focus:outline-none"
            style={{ fieldSizing: 'content' } as CSSProperties}
          />
          <div className="mb-1 hidden items-center gap-2 px-1 sm:flex">
            <StrategyTag strategy={strategy} />
            <span className="font-mono text-xs text-muted">k={k}</span>
          </div>
          {busy ? (
            <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()}>
              <StopIcon size={16} /> Stop
            </Button>
          ) : (
            <Button type="submit" disabled={!input.trim()}>
              <SendIcon size={16} /> Send
            </Button>
          )}
        </div>
      </form>
    </div>
  )
}

function Setting({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="label-caps mb-2">{label}</div>
      {children}
    </div>
  )
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="pt-10 text-center sm:pt-20">
      <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl border border-line-strong bg-raised glow">
        <SparkIcon size={26} className="text-accent" />
      </div>
      <h1 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">
        Ask the <span className="text-gradient">biomedical literature</span>
      </h1>
      <p className="mx-auto mt-3 max-w-xl text-ink-2">
        Answers come only from 27,972 PubMed abstracts, retrieved locally and cited inline. Every
        model runs on this machine.
      </p>
      <div className="mx-auto mt-8 grid max-w-2xl gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="panel p-3 text-left text-sm text-ink-2 transition hover:border-accent hover:text-ink"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

function TurnView({ turn }: { turn: Turn }) {
  const cited = turn.stage === 'done' ? citedNumbers(turn.answer) : undefined
  return (
    <article className="space-y-4">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md border border-accent/30 bg-accent/10 px-4 py-2.5 text-ink">
          {turn.question}
        </div>
      </div>

      <div className="panel p-4 sm:p-5">
        {turn.stage === 'retrieving' && (
          <div className="scanbar rounded-md py-1 text-sm text-muted">Retrieving sources…</div>
        )}
        {turn.stage === 'generating' && !turn.answer && (
          <div className="scanbar rounded-md py-1 text-sm text-muted">
            Generating answer from {turn.sources.length} sources…
          </div>
        )}
        {turn.answer && (
          <Answer text={turn.answer} sources={turn.sources} streaming={turn.stage === 'generating'} />
        )}
        {turn.stage === 'stopped' && <p className="mt-2 text-xs text-muted">Stopped.</p>}
        {turn.error && (
          <p className="mt-2 text-sm text-critical" role="alert">
            {turn.error}
          </p>
        )}

        {turn.sources.length > 0 && (
          <div className="mt-5 border-t border-line pt-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <span className="label-caps">Sources</span>
              {turn.done && (
                <span className="font-mono text-[0.7rem] text-muted">
                  {turn.done.model} · {turn.done.output_tokens} tokens · {turn.done.seconds}s
                </span>
              )}
            </div>
            <SourceList sources={turn.sources} cited={cited} />
          </div>
        )}
      </div>
    </article>
  )
}

