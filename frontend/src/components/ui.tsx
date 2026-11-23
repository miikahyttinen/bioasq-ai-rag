import type { ButtonHTMLAttributes, ReactNode } from 'react'
import type { Strategy } from '../api/types'
import { STRATEGY_META } from '../lib/strategies'
import { cx } from '../lib/format'
import { AlertIcon } from './icons'

export function Panel({
  title,
  actions,
  className,
  children,
}: {
  title?: ReactNode
  actions?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <section className={cx('panel p-4 sm:p-5', className)}>
      {(title || actions) && (
        <header className="mb-4 flex items-center justify-between gap-3">
          {title && <h2 className="label-caps">{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'ghost' | 'outline'
  size?: 'sm' | 'md'
}

export function Button({
  variant = 'primary',
  size = 'md',
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition',
        'disabled:cursor-not-allowed disabled:opacity-40',
        size === 'sm' ? 'px-2.5 py-1.5 text-xs' : 'px-4 py-2 text-sm',
        variant === 'primary' &&
          'bg-accent text-on-accent shadow-[0_0_20px_-4px_var(--accent-glow)] hover:brightness-110',
        variant === 'outline' &&
          'border border-line-strong text-ink-2 hover:border-accent hover:text-ink',
        variant === 'ghost' && 'text-ink-2 hover:bg-raised hover:text-ink',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: { value: T; label: ReactNode }[]
  onChange: (value: T) => void
  label: string
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex rounded-lg border border-line bg-surface p-0.5"
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded-md px-2.5 py-1 text-xs font-medium transition',
            o.value === value ? 'bg-raised text-ink shadow-sm glow' : 'text-muted hover:text-ink',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function StrategyTag({ strategy }: { strategy: Strategy }) {
  const meta = STRATEGY_META[strategy]
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-2">
      <span className={cx('h-2 w-2 rounded-full', meta.dot)} aria-hidden="true" />
      {meta.label}
    </span>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-muted" role="status">
      <span className="relative flex h-2.5 w-2.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-accent" />
      </span>
      {label}
    </div>
  )
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className="flex items-start gap-2 rounded-lg border border-critical/40 bg-critical/10 p-3 text-sm text-ink">
      <AlertIcon size={16} className="mt-0.5 shrink-0 text-critical" />
      <span>{message}</span>
    </div>
  )
}

export function StatTile({
  label,
  value,
  hint,
  accent = false,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  accent?: boolean
}) {
  return (
    <div className={cx('panel p-4', accent && 'glow')}>
      <div className="label-caps">{label}</div>
      <div className="mt-2 font-display text-3xl font-semibold tracking-tight text-ink">
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </div>
  )
}

/** 1-5 judge score with a numeric label (never color alone). */
export function ScoreBadge({ score, label }: { score: number | null | undefined; label: string }) {
  if (score == null) return <span className="text-xs text-muted">{label}: —</span>
  const tone =
    score >= 4
      ? 'border-good/40 text-good'
      : score === 3
        ? 'border-warning/40 text-warning'
        : 'border-critical/40 text-critical'
  return (
    <span
      className={cx('inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs', tone)}
    >
      <span className="text-muted">{label}</span>
      <span className="font-mono font-semibold">{score}/5</span>
    </span>
  )
}
