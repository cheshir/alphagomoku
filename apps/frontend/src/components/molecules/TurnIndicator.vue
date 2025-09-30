<template>
  <div :class="['turn-indicator', statusClass]">
    <div class="turn-indicator__content">
      <BaseSpinner v-if="isLoading" class="turn-indicator__spinner" />
      <span class="turn-indicator__text">{{ displayText }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { BaseSpinner } from '@/components/atoms'

interface Props {
  isPlayerTurn: boolean
  isAiThinking: boolean
  isGameOver: boolean
  winner?: string | null
}

const props = defineProps<Props>()

const statusClass = computed(() => {
  if (props.isGameOver) return 'turn-indicator--game-over'
  if (props.isAiThinking) return 'turn-indicator--ai-thinking'
  if (props.isPlayerTurn) return 'turn-indicator--player-turn'
  return 'turn-indicator--ai-turn'
})

const isLoading = computed(() => props.isAiThinking)

const displayText = computed(() => {
  if (props.isGameOver) return props.winner || 'Game Over'
  if (props.isAiThinking) return 'AI is thinking...'
  if (props.isPlayerTurn) return 'Your turn'
  return "AI's turn"
})
</script>

<style scoped>
.turn-indicator {
  padding: 16px 24px;
  border-radius: 12px;
  text-align: center;
  font-weight: 600;
  font-size: 16px;
  transition: all 0.3s ease;
}

.turn-indicator__content {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.turn-indicator__spinner {
  color: currentColor;
}

.turn-indicator__text {
  font-weight: 600;
}

.turn-indicator--player-turn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  animation: pulse 2s ease-in-out infinite;
}

.turn-indicator--ai-turn {
  background: #e9ecef;
  color: #6c757d;
}

.turn-indicator--ai-thinking {
  background: #fff3cd;
  color: #856404;
}

.turn-indicator--game-over {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  font-size: 18px;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.02);
  }
}
</style>
