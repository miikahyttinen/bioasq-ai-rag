import type {
  ChatDone,
  ChatMessage,
  Cluster,
  EvalResults,
  Health,
  Passage,
  QuestionDetail,
  QuestionPage,
  QuestionRun,
  SearchResponse,
  Source,
  Strategy,
  TopicPoint,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string, params?: Record<string, string | number | boolean>) {
  const query = params
    ? '?' + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))
    : ''
  const response = await fetch(`/api${path}${query}`)
  if (!response.ok) throw await toError(response)
  return (await response.json()) as T
}

async function toError(response: Response) {
  let detail = response.statusText
  try {
    const body = await response.json()
    detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
  } catch {
    // non-JSON error body
  }
  return new ApiError(response.status, detail || `HTTP ${response.status}`)
}

export const api = {
  health: () => get<Health>('/health'),
  search: (q: string, strategy: Strategy, k: number) =>
    get<SearchResponse>('/search', { q, strategy, k }),
  passage: (pmid: number) => get<Passage>(`/passages/${pmid}`),
  questions: (query: string, page: number, pageSize: number, usableOnly: boolean) =>
    get<QuestionPage>('/questions', {
      query,
      page,
      page_size: pageSize,
      usable_only: usableOnly,
    }),
  question: (qid: number) => get<QuestionDetail>(`/questions/${qid}`),
  runQuestion: (qid: number, strategy: Strategy, k: number, model?: string) =>
    get<QuestionRun>(`/questions/${qid}/run`, { strategy, k, ...(model ? { model } : {}) }),
  topicMap: () => get<TopicPoint[]>('/topics/map'),
  clusters: () => get<Cluster[]>('/topics/clusters'),
  evalResults: () => get<EvalResults>('/eval/results'),
}

export interface ChatRequest {
  message: string
  history: ChatMessage[]
  strategy: Strategy
  k: number
  model?: string
  think?: boolean
}

export interface ChatHandlers {
  onSources: (sources: Source[]) => void
  onToken: (text: string) => void
  onDone: (done: ChatDone) => void
  onError: (message: string) => void
}

/** POST /api/chat and dispatch its Server-Sent Events. Resolves when the stream ends. */
export async function streamChat(
  request: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok || !response.body) throw await toError(response)

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      dispatch(buffer.slice(0, boundary), handlers)
      buffer = buffer.slice(boundary + 2)
    }
  }
  if (buffer.trim()) dispatch(buffer, handlers)
}

function dispatch(block: string, handlers: ChatHandlers) {
  let event = 'message'
  let data = ''
  for (const line of block.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7)
    else if (line.startsWith('data: ')) data += line.slice(6)
  }
  if (!data) return
  const payload = JSON.parse(data)
  if (event === 'sources') handlers.onSources(payload.sources)
  else if (event === 'token') handlers.onToken(payload.text)
  else if (event === 'done') handlers.onDone(payload)
  else if (event === 'error') handlers.onError(payload.message)
}
