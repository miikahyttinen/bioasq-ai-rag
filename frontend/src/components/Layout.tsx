import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { api } from '../api/client'
import { useTheme } from '../lib/theme'
import {
  AlertIcon,
  ChartIcon,
  ChatIcon,
  ListIcon,
  MapIcon,
  MoonIcon,
  SearchIcon,
  SunIcon,
} from './icons'
import { cx } from '../lib/format'

const NAV = [
  { to: '/', label: 'Chat', icon: ChatIcon, end: true },
  { to: '/search', label: 'Search', icon: SearchIcon },
  { to: '/questions', label: 'Questions', icon: ListIcon },
  { to: '/topics', label: 'Topic Map', icon: MapIcon },
  { to: '/evaluation', label: 'Evaluation', icon: ChartIcon },
]

export function Layout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme()
  const health = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: false,
  })

  const online = health.data?.status === 'ok' || health.data?.status === 'degraded'

  return (
    <div className="flex min-h-full flex-col md:flex-row">
      <aside className="z-20 flex shrink-0 items-center gap-1 border-b border-line bg-panel px-3 py-2 backdrop-blur md:sticky md:top-0 md:h-screen md:w-60 md:flex-col md:items-stretch md:gap-0 md:border-r md:border-b-0 md:px-4 md:py-6">
        <div className="mr-2 md:mr-0 md:mb-8 md:px-2">
          <div className="font-display text-lg font-semibold tracking-tight">
            <span className="text-gradient">BioASQ</span>
            <span className="text-ink"> RAG</span>
          </div>
          <div className="label-caps hidden md:block">Local explorer</div>
        </div>

        <nav className="flex flex-1 gap-1 overflow-x-auto md:flex-col md:overflow-visible">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cx(
                  'group relative flex shrink-0 items-center gap-3 rounded-lg px-3 py-2 text-sm transition',
                  isActive
                    ? 'bg-raised text-ink glow'
                    : 'text-ink-2 hover:bg-raised/60 hover:text-ink',
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Icon className={isActive ? 'text-accent' : 'text-muted group-hover:text-ink'} />
                  <span className="hidden sm:inline">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-2 md:mt-6 md:flex-col md:items-stretch">
          <div
            className="hidden items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs md:flex"
            title={health.data?.problems.join('\n')}
          >
            <span
              className={cx(
                'h-2 w-2 rounded-full',
                online ? 'bg-good shadow-[0_0_8px_var(--good)]' : 'bg-critical',
              )}
              aria-hidden="true"
            />
            <span className="text-ink-2">
              {health.isLoading ? 'Checking…' : online ? 'Systems online' : 'Offline'}
            </span>
          </div>
          <button
            onClick={toggle}
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-ink-2 hover:bg-raised hover:text-ink"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <SunIcon size={16} /> : <MoonIcon size={16} />}
            <span className="hidden md:inline">{theme === 'dark' ? 'Light' : 'Dark'} mode</span>
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <HealthBanner
          error={health.error}
          status={health.data?.status}
          problems={health.data?.problems}
        />
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {children}
        </main>
        <footer className="px-6 pb-4 text-center text-[0.7rem] text-muted">
          Research & study tool, not medical advice · Data: rag-mini-bioasq (CC BY 2.5, derived
          from BioASQ / PubMed)
        </footer>
      </div>
    </div>
  )
}

function HealthBanner({
  error,
  status,
  problems,
}: {
  error: unknown
  status?: 'ok' | 'degraded' | 'down'
  problems?: string[]
}) {
  if (!error && status !== 'down') return null
  const message = error
    ? 'Cannot reach the backend. Start it with: cd backend && uv run uvicorn app.main:app --port 8000'
    : problems?.join(' · ') || 'The backend reports it is down.'
  return (
    <div
      role="alert"
      className="flex items-start gap-3 border-b border-critical/40 bg-critical/10 px-6 py-3 text-sm"
    >
      <AlertIcon size={18} className="mt-0.5 shrink-0 text-critical" />
      <div>
        <div className="font-medium text-ink">
          {error ? 'Backend offline' : 'Ollama or indexes unavailable'}
        </div>
        <div className="text-ink-2">{message}</div>
        {!error && (
          <div className="mt-1 text-xs text-muted">
            Start Ollama with: brew services start ollama
          </div>
        )}
      </div>
    </div>
  )
}
