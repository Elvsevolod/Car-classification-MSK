export type Box = { x: number; y: number; w: number; h: number }
export type Query = Box & { image_id: string }
export type Candidate = Box & { rank: number; image_id: string; similarity: number; rerank_score: number; crop_url: string }
export type SearchMode = 'ranking' | 'candidates'
export type SearchResult = { mode: SearchMode; threshold_source: string | null; confidence: number; elapsed_ms: number; refused: boolean; results: Candidate[]; accepted_candidate: Candidate | null; threshold: number | null; profile: string; profile_fingerprint: string; ranking_policy: string; candidate_policy: string; demo_threshold_override: boolean }
export type Health = { status: string; device: string; gallery_size: number; model: string; default_threshold: number | null; profile: string; profile_fingerprint: string; embedding_dim: number; ranking_policy: string; candidate_policy: string }
export const boxKeys: (keyof Box)[] = ['x', 'y', 'w', 'h']

export type ModelMetrics = {
  model: string
  threshold: number
  validation: {
    mAP_at_10: number
    candidate_F1: number
    TNR: number
    known_queries?: number
    unknown_queries?: number
  }
}
