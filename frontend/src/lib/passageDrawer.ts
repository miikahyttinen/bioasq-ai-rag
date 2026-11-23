import { createContext, useContext } from 'react'

export interface Opened {
  pmid: number
  /** The retrieved chunk to highlight inside the full abstract, if any. */
  highlight?: string
  /** Citation number in the current answer, if opened from one. */
  n?: number
}

export const PassageContext = createContext<(opened: Opened) => void>(() => {})

/** Open the passage drawer for a PubMed ID from anywhere in the app. */
export function usePassageDrawer() {
  return useContext(PassageContext)
}
