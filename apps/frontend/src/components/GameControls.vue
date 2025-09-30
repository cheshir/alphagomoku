<template>
  <div class="game-controls">
    <NewGameForm
      v-if="!gameId"
      @start-game="handleStartGame"
    />

    <ActiveGamePanel
      v-else
      :difficulty="difficulty"
      :player-color="playerColor"
      :move-count="moveCount"
      :is-player-turn="isPlayerTurn"
      :is-game-over="isGameOver"
      :is-ai-thinking="isAiThinking"
      :winner="winner"
      @resign="$emit('resign')"
      @reset-to-setup="$emit('resetToSetup')"
    />
  </div>
</template>

<script setup lang="ts">
import { NewGameForm, ActiveGamePanel } from '@/components/organisms'
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

defineProps<Props>()

const emit = defineEmits<{
  newGame: [difficulty: Difficulty, color: Player]
  resign: []
  resetToSetup: []
}>()

function handleStartGame(difficulty: Difficulty, color: Player) {
  emit('newGame', difficulty, color)
}
</script>

<style scoped>
.game-controls {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  min-width: 320px;
}
</style>