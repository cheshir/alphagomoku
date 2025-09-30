<template>
  <div class="active-game-panel">
    <GameStatusCard
      :difficulty="difficulty"
      :player-color="playerColor"
      :move-count="moveCount"
    />

    <TurnIndicator
      :is-player-turn="isPlayerTurn"
      :is-ai-thinking="isAiThinking"
      :is-game-over="isGameOver"
      :winner="winner"
    />

    <GameActions
      :show-resign="!isGameOver"
      :disabled="isAiThinking"
      @resign="$emit('resign')"
      @new-game="$emit('resetToSetup')"
    />
  </div>
</template>

<script setup lang="ts">
import { GameStatusCard, TurnIndicator, GameActions } from '@/components/molecules'
import type { Difficulty, Player } from '@/types/game'

interface Props {
  difficulty: Difficulty
  playerColor: Player
  moveCount: number
  isPlayerTurn: boolean
  isGameOver: boolean
  isAiThinking: boolean
  winner: string | null
}

defineProps<Props>()

defineEmits<{
  resign: []
  resetToSetup: []
}>()
</script>

<style scoped>
.active-game-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 24px;
}
</style>
