import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { PassageProvider } from './components/PassageDrawer'
import { Spinner } from './components/ui'
import ChatPage from './pages/ChatPage'
import ExplorerPage from './pages/ExplorerPage'
import SearchPage from './pages/SearchPage'

// Plotly (~4.5 MB) and Recharts load only when their pages are opened.
const TopicMapPage = lazy(() => import('./pages/TopicMapPage'))
const EvaluationPage = lazy(() => import('./pages/EvaluationPage'))

function App() {
  return (
    <PassageProvider>
      <Layout>
        <Suspense fallback={<Spinner label="Loading page" />}>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/questions" element={<ExplorerPage />} />
            <Route path="/topics" element={<TopicMapPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
          </Routes>
        </Suspense>
      </Layout>
    </PassageProvider>
  )
}

export default App
