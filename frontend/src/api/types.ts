// Mirrors backend/app/models.py and the /api responses.

export type Strategy = 'dense' | 'bm25' | 'hybrid'
export const STRATEGIES: Strategy[] = ['dense', 'bm25', 'hybrid']

export interface Source {
  n: number
  chunk_id: string
  pmid: number
  text: string
  score: number
  url: string
  cluster_id: number | null
  cluster_label: string | null
}

export interface RunSource extends Source {
  in_gold: boolean
}

export interface SearchResponse {
  query: string
  strategy: Strategy
  k: number
  seconds: number
  results: Source[]
}

export interface Passage {
  pmid: number
  text: string
  url: string
  cluster_id: number | null
  cluster_label: string | null
  nearest_cluster_id: number | null
  nearest_cluster_label: string | null
}

export interface QuestionSummary {
  qid: number
  question: string
  n_gold: number
  has_usable: boolean
}

export interface QuestionPage {
  total: number
  page: number
  page_size: number
  items: QuestionSummary[]
}

export interface RetrievalMetrics {
  'recall@1': number
  'recall@5': number
  'recall@10': number
  'recall@20': number
  'hit@10': number
  mrr: number
}

export interface Judge {
  ok: boolean
  reason?: string
  correctness?: number
  groundedness?: number | null
}

export interface AnswerRecord {
  qid: number
  hits: number
  retrieved: { pmid: number; chunk_id: string; score: number; in_gold: boolean }[]
  rag: { answer: string; seconds: number; judge: Judge }
  no_rag: { answer: string; seconds: number; judge: Judge }
}

export interface QuestionDetail extends QuestionSummary {
  answer: string
  relevant_ids: number[]
  usable_relevant_ids: number[]
  retrieval: Record<Strategy, RetrievalMetrics & { pmids: number[] }> | null
  evaluation: AnswerRecord | null
}

export interface QuestionRun {
  qid: number
  question: string
  gold_answer: string
  gold_ids: number[]
  strategy: Strategy
  k: number
  sources: RunSource[]
  hits: number
  recall: number | null
  answer: string
  model: string
  seconds: number
}

export interface TopicPoint {
  pmid: number
  x: number
  y: number
  cluster_id: number
  nearest_cluster: number
}

export interface Cluster {
  cluster_id: number
  label: string
  description: string
  size: number
  x: number | null
  y: number | null
  representative_pmids: number[]
}

export interface AnswerSide {
  correctness: number | null
  groundedness?: number | null
  correctness_distribution: Record<string, number>
  share_correct_4plus: number
  judge_failures: number
  mean_answer_seconds: number
}

export interface EvalResults {
  retrieval: {
    n_questions: number
    depth: number
    embed_model: string
    best_strategy: Strategy
    strategies: Record<Strategy, RetrievalMetrics & { seconds: number }>
  } | null
  answers: {
    n_questions: number
    config: { model: string; strategy: Strategy; k: number; seed: number }
    rag: AnswerSide
    no_rag: AnswerSide
    rag_correctness_by_retrieval: Record<
      'with_gold_source' | 'without_gold_source',
      { n: number; correctness: number | null }
    >
    seconds: number
    note: string
  } | null
  answer_records: {
    qid: number
    question: string
    gold_answer: string
    hits: number
    n_sources: number
    rag: { answer: string; correctness: number | null; groundedness: number | null; reason: string }
    no_rag: { answer: string; correctness: number | null; reason: string }
  }[]
  worst_retrieval: {
    strategy: Strategy | null
    questions: ({ qid: number; question: string; n_gold: number } & RetrievalMetrics)[]
  }
}

export interface Health {
  status: 'ok' | 'degraded' | 'down'
  ollama: boolean
  chat_model: string
  models: Record<string, boolean>
  collection_count: number | null
  bm25_index: boolean
  topics: boolean
  eval_results: boolean
  problems: string[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatDone {
  sources: Source[]
  model: string
  prompt_tokens: number
  output_tokens: number
  seconds: number
}
