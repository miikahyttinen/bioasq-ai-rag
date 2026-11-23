export function cx(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}

export function pct(value: number | null | undefined, digits = 0) {
  return value == null ? '—' : `${(value * 100).toFixed(digits)}%`
}
