import type { Strategy } from '../api/types'

export const STRATEGY_META: Record<
  Strategy,
  { label: string; short: string; blurb: string; cssVar: string; dot: string }
> = {
  dense: {
    label: 'Dense',
    short: 'D',
    blurb: 'Meaning-based vector search (qwen3-embedding)',
    cssVar: '--series-dense',
    dot: 'bg-dense',
  },
  bm25: {
    label: 'BM25',
    short: 'B',
    blurb: 'Keyword search with stemming',
    cssVar: '--series-bm25',
    dot: 'bg-bm25',
  },
  hybrid: {
    label: 'Hybrid',
    short: 'H',
    blurb: 'Dense + BM25 merged with Reciprocal Rank Fusion',
    cssVar: '--series-hybrid',
    dot: 'bg-hybrid',
  },
}

export const MODELS = [
  { id: 'gemma4:26b-a4b-it-qat', label: 'Gemma 4 26B', note: 'default · ~58 tok/s' },
  { id: 'gemma4:12b-it-qat', label: 'Gemma 4 12B', note: 'low memory · ~28 tok/s' },
]
