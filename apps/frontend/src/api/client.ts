import axios from 'axios'
import type {
  CreateGameRequest,
  CreateGameResponse,
  GameState,
  MoveRequest,
  MoveResponse
} from '@/types/game'

const API_BASE_URL = import.meta.env.VITE_API_URL

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

export const gameApi = {
  async createGame(request: CreateGameRequest): Promise<CreateGameResponse> {
    const { data } = await client.post<CreateGameResponse>('/api/games', request)
    return data
  },

  async getGameState(gameId: string): Promise<GameState> {
    const { data } = await client.get<GameState>(`/api/games/${gameId}`)
    return data
  },

  async makeMove(gameId: string, move: MoveRequest): Promise<MoveResponse> {
    const { data } = await client.post<MoveResponse>(`/api/games/${gameId}/move`, move)
    return data
  },

  async resignGame(gameId: string): Promise<{ game_id: string; status: string; winner: number }> {
    const { data } = await client.post(`/api/games/${gameId}/resign`)
    return data
  },

  async deleteGame(gameId: string): Promise<void> {
    await client.delete(`/api/games/${gameId}`)
  },

  async healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
    const { data } = await client.get('/health')
    return data
  }
}

export default client