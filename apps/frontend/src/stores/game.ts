import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { gameApi } from '@/api/client'
import type {
  Difficulty,
  Player,
  GameStatus,
  Move,
  DebugInfo
} from '@/types/game'

export const useGameStore = defineStore('game', () => {
  // State
  const gameId = ref<string | null>(null)
  const board = ref<number[][]>([])
  const boardSize = ref(15)
  const difficulty = ref<Difficulty>('medium')
  const playerColor = ref<Player>(1)
  const currentPlayer = ref<Player>(1)
  const status = ref<GameStatus>('in_progress')
  const lastMove = ref<Move | null>(null)
  const moveCount = ref(0)
  const playerTime = ref(0)
  const aiTime = ref(0)
  const debugInfo = ref<DebugInfo | null>(null)
  const isAiThinking = ref(false)
  const error = ref<string | null>(null)

  // Computed
  const isPlayerTurn = computed(() => currentPlayer.value === playerColor.value)
  const isGameOver = computed(() => status.value !== 'in_progress')
  const winner = computed(() => {
    if (status.value === 'player_won') return 'You won!'
    if (status.value === 'ai_won') return 'AI won!'
    if (status.value === 'draw') return 'Draw!'
    return null
  })

  // Actions
  async function createGame(selectedDifficulty: Difficulty, selectedColor: Player) {
    try {
      error.value = null
      const response = await gameApi.createGame({
        difficulty: selectedDifficulty,
        player_color: selectedColor
      })

      gameId.value = response.game_id
      difficulty.value = selectedDifficulty
      playerColor.value = selectedColor
      currentPlayer.value = response.current_player
      status.value = response.status
      boardSize.value = response.board_size
      board.value = Array(boardSize.value).fill(null).map(() => Array(boardSize.value).fill(0))
      moveCount.value = 0
      playerTime.value = 0
      aiTime.value = 0
      lastMove.value = null
      debugInfo.value = response.debug_info

      // If AI made the first move
      if (response.ai_move) {
        board.value[response.ai_move.row][response.ai_move.col] = response.ai_move.player
        lastMove.value = response.ai_move
        moveCount.value = 1
      }
    } catch (e: any) {
      error.value = e.response?.data?.detail || 'Failed to create game'
      throw e
    }
  }

  async function makeMove(row: number, col: number) {
    if (!gameId.value || isGameOver.value || !isPlayerTurn.value || isAiThinking.value) {
      return
    }

    if (board.value[row][col] !== 0) {
      error.value = 'Position already occupied'
      return
    }

    try {
      error.value = null
      isAiThinking.value = true

      const response = await gameApi.makeMove(gameId.value, { row, col })

      // Update board with player move
      board.value[response.player_move.row][response.player_move.col] = response.player_move.player

      // Update board with AI move if present
      if (response.ai_move) {
        board.value[response.ai_move.row][response.ai_move.col] = response.ai_move.player
        lastMove.value = response.ai_move
      } else {
        lastMove.value = response.player_move
      }

      currentPlayer.value = response.current_player
      status.value = response.status
      moveCount.value = response.move_count
      playerTime.value = response.player_time
      aiTime.value = response.ai_time
      debugInfo.value = response.debug_info
    } catch (e: any) {
      error.value = e.response?.data?.detail || 'Failed to make move'
      throw e
    } finally {
      isAiThinking.value = false
    }
  }

  async function resign() {
    if (!gameId.value || isGameOver.value) return

    try {
      error.value = null
      await gameApi.resignGame(gameId.value)
      status.value = 'ai_won'
    } catch (e: any) {
      error.value = e.response?.data?.detail || 'Failed to resign'
      throw e
    }
  }

  function resetGame() {
    gameId.value = null
    board.value = []
    currentPlayer.value = 1
    status.value = 'in_progress'
    lastMove.value = null
    moveCount.value = 0
    playerTime.value = 0
    aiTime.value = 0
    debugInfo.value = null
    isAiThinking.value = false
    error.value = null
  }

  return {
    // State
    gameId,
    board,
    boardSize,
    difficulty,
    playerColor,
    currentPlayer,
    status,
    lastMove,
    moveCount,
    playerTime,
    aiTime,
    debugInfo,
    isAiThinking,
    error,
    // Computed
    isPlayerTurn,
    isGameOver,
    winner,
    // Actions
    createGame,
    makeMove,
    resign,
    resetGame
  }
})