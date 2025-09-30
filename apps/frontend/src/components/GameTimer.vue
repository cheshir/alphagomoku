<template>
  <div class="game-timer">
    <div class="timer-card" :class="{ active: isPlayerTurn && !isGameOver }">
      <div class="timer-label">
        <div class="player-indicator black"></div>
        <span>{{ playerColor === 1 ? 'You' : 'AI' }}</span>
      </div>
      <div class="timer-value">{{ formatTime(playerColor === 1 ? playerTime : aiTime) }}</div>
    </div>

    <div class="timer-card" :class="{ active: !isPlayerTurn && !isGameOver && !isAiThinking }">
      <div class="timer-label">
        <div class="player-indicator white"></div>
        <span>{{ playerColor === -1 ? 'You' : 'AI' }}</span>
      </div>
      <div class="timer-value">{{ formatTime(playerColor === -1 ? playerTime : aiTime) }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Player } from '@/types/game'

interface Props {
  playerTime: number
  aiTime: number
  playerColor: Player
  isPlayerTurn: boolean
  isGameOver: boolean
  isAiThinking: boolean
}

defineProps<Props>()

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}
</script>

<style scoped>
.game-timer {
  display: flex;
  gap: 16px;
  justify-content: center;
  margin-bottom: 20px;
}

.timer-card {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 16px 24px;
  min-width: 140px;
  text-align: center;
  border: 2px solid transparent;
  transition: all 0.3s;
}

.timer-card.active {
  background: #e3f2fd;
  border-color: #4a90e2;
  box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
}

.timer-label {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #666;
}

.player-indicator {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 1px solid #333;
}

.player-indicator.black {
  background: linear-gradient(135deg, #2a2a2a 0%, #000000 100%);
}

.player-indicator.white {
  background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
}

.timer-value {
  font-size: 28px;
  font-weight: 700;
  font-family: 'Courier New', monospace;
  color: #333;
}
</style>