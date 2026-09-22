export type Box = { x: number; y: number; w: number; h: number }
export type Query = Box & { image_id: string }
export type Candidate = Box & { rank: number; image_id: string; similarity: number; rerank_score: number; crop_url: string }
export type SearchResult = { confidence: number; elapsed_ms: number; refused: boolean; results: Candidate[]; threshold: number | null }
export type Health = { device: string; gallery_size: number; model: string }
export const boxKeys: (keyof Box)[] = ['x', 'y', 'w', 'h']
