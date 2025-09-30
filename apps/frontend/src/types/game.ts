export type Difficulty = 'easy' | 'medium' | 'hard'
export type GameStatus = 'in_progress' | 'player_won' | 'ai_won' | 'draw'
export type Player = 1 | -1

export interface Move {
  row: number
  col: number
  player: Player
}

export interface TopMove {
  row: number
  col: number
  probability: number
  visits: number
}

export interface DebugInfo {
  policy_distribution: number[][] | null
  value_estimate: number
  simulations: number
  thinking_time_ms: number
  top_moves: TopMove[]
  search_depth: number
  nodes_explored: number
}

export interface GameState {
  game_id: string
  board: number[][]
  board_size: number
  difficulty: Difficulty
  player_color: Player
  current_player: Player
  status: GameStatus
  last_move: Move | null
  move_count: number
  player_time: number
  ai_time: number
}

export interface CreateGameRequest {
  difficulty: Difficulty
  player_color: Player
}

export interface CreateGameResponse extends Omit<GameState, 'board' | 'last_move' | 'move_count' | 'player_time' | 'ai_time'> {
  ai_move: Move | null
  debug_info: DebugInfo | null
}

export interface MoveRequest {
  row: number
  col: number
}

export interface MoveResponse {
  player_move: Move
  ai_move: Move | null
  board: number[][]
  current_player: Player
  status: GameStatus
  move_count: number
  player_time: number
  ai_time: number
  debug_info: DebugInfo | null
}