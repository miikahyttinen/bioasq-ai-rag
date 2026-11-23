/**
 * Turn inline citations like "[1]", "[1, 3]" or "[2-4]" into markdown links
 * "[1](#cite-1)" so the renderer can show them as clickable chips.
 */
const CITATION = /\[(\d+(?:\s*(?:,|–|-)\s*\d+)*)\](?!\()/g

export function expandCitation(inner: string): number[] {
  const numbers: number[] = []
  for (const part of inner.split(',')) {
    const range = part.split(/[–-]/).map((s) => parseInt(s.trim(), 10))
    if (range.length === 2 && range[0] <= range[1] && range[1] - range[0] < 20) {
      for (let n = range[0]; n <= range[1]; n++) numbers.push(n)
    } else if (!Number.isNaN(range[0])) {
      numbers.push(range[0])
    }
  }
  return numbers
}

export function linkCitations(markdown: string, maxSource: number): string {
  return markdown.replace(CITATION, (match, inner: string) => {
    const numbers = expandCitation(inner)
    if (!numbers.length || numbers.some((n) => n < 1 || n > maxSource)) return match
    return numbers.map((n) => `[${n}](#cite-${n})`).join('')
  })
}

export function citedNumbers(markdown: string): Set<number> {
  const cited = new Set<number>()
  for (const match of markdown.matchAll(CITATION)) {
    expandCitation(match[1]).forEach((n) => cited.add(n))
  }
  return cited
}
