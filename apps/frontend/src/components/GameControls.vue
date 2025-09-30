<template>
  <div class="game-controls">
    <div v-if="!gameId" class="new-game-section">
      <h2>New Game</h2>
      <div class="control-group">
        <label>Difficulty:</label>
        <select v-model="selectedDifficulty" class="select-input">
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>
      <div class="control-group">
        <label>Play as:</label>
        <select v-model="selectedColor" class="select-input">
          <option :value="1">Black (First)</option>
          <option :value="-1">White (Second)</option>
        </select>
      </div>
      <button @click="handleNewGame" class="btn btn-primary">
        Start Game
      </button>
    </div>

    <div v-else class="game-info-section">
      <div class="status-card">
        <h3 class="status-title">Game Status</h3>
        <div class="status-content">
          <div class="status-item">
            <span class="label">Difficulty:</span>
            <span class="value">{{ difficultyLabel }}</span>
          </div>
          <div class="status-item">
            <span class="label">You are:</span>
            <span class="value">{{ playerColor === 1 ? 'Black' : 'White' }}</span>
          </div>
          <div class="status-item">
            <span class="label">Move:</span>
            <span class="value">{{ moveCount }}</span>
          </div>
        </div>
      </div>

      <div class="turn-indicator" :class="turnIndicatorClass">
        <div class="turn-content">
          <div v-if="isGameOver" class="game-over">
            <div class="winner-text">{{ winner }}</div>
          </div>
          <div v-else-if="isAiThinking" class="ai-thinking">
            <div class="spinner"></div>
            <span>AI is thinking...</span>
          </div>
          <div v-else-if="isPlayerTurn" class="your-turn">
            Your turn
          </div>
          <div v-else class="ai-turn">
            AI's turn
          </div>
        </div>
      </div>

      <div class="action-buttons">
        <button
          v-if="!isGameOver"
          @click="handleResign"
          class="btn btn-secondary"
          :disabled="isAiThinking"
        >
          Resign
        </button>
        <button @click="handleNewGameClick" class="btn btn-primary">
          New Game
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Difficulty, Player } from '@/types/game'

interface Props {
  gameId: string | null
  difficulty: Difficulty
  playerColor: Player
  moveCount: number
  isPlayerTurn: boolean
  isGameOver: boolean
  isAiThinking: boolean
  winner: string | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  newGame: [difficulty: Difficulty, color: Player]
  resign: []
  resetToSetup: []
}>()

const selectedDifficulty = ref<Difficulty>('medium')
const selectedColor = ref<Player>(1)

const difficultyLabel = computed(() => {
  const labels = {
    easy: 'Easy',
    medium: 'Medium',
    hard: 'Hard'
  }
  return labels[props.difficulty]
})

const turnIndicatorClass = computed(() => {
  if (props.isGameOver) return 'game-over-state'
  if (props.isAiThinking) return 'ai-thinking-state'
  if (props.isPlayerTurn) return 'player-turn-state'
  return 'ai-turn-state'
})

function handleNewGameClick() {
  emit('resetToSetup')
}

function handleNewGame() {
  emit('newGame', selectedDifficulty.value, selectedColor.value)
}

function handleResign() {
  if (confirm('Are you sure you want to resign?')) {
    emit('resign')
  }
}
</script>

<style scoped>
.game-controls {
  padding: 24px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  min-width: 320px;
}

.new-game-section h2 {
  margin: 0 0 20px 0;
  font-size: 24px;
  font-weight: 600;
  color: #333;
}

.control-group {
  margin-bottom: 16px;
}

.control-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #555;
}

.select-input {
  width: 100%;
  padding: 10px 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 14px;
  background: white;
  cursor: pointer;
  transition: border-color 0.2s;
}

.select-input:hover {
  border-color: #c0c0c0;
}

.select-input:focus {
  outline: none;
  border-color: #4a90e2;
}

.btn {
  width: 100%;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #4a90e2;
  color: white;
  margin-top: 8px;
}

.btn-primary:hover:not(:disabled) {
  background: #357abd;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(74, 144, 226, 0.3);
}

.btn-secondary {
  background: #e74c3c;
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  background: #c0392b;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(231, 76, 60, 0.3);
}

.game-info-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.status-card {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 16px;
}

.status-title {
  margin: 0 0 12px 0;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.status-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status-item .label {
  color: #666;
  font-size: 14px;
}

.status-item .value {
  font-weight: 600;
  color: #333;
  font-size: 14px;
}

.turn-indicator {
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  font-weight: 600;
  transition: all 0.3s;
}

.player-turn-state {
  background: #d4edda;
  color: #155724;
  border: 2px solid #c3e6cb;
}

.ai-turn-state {
  background: #fff3cd;
  color: #856404;
  border: 2px solid #ffeaa7;
}

.ai-thinking-state {
  background: #cce5ff;
  color: #004085;
  border: 2px solid #b8daff;
}

.game-over-state {
  background: #f8d7da;
  color: #721c24;
  border: 2px solid #f5c6cb;
}

.turn-content {
  font-size: 16px;
}

.ai-thinking {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.spinner {
  width: 20px;
  height: 20px;
  border: 3px solid #004085;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.winner-text {
  font-size: 20px;
  font-weight: 700;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
</style>