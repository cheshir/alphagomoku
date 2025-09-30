<template>
  <BaseCard>
    <h3 class="status-title">Game Status</h3>
    <div class="status-content">
      <StatusItem label="Difficulty" :value="difficulty" />
      <StatusItem label="You are" :value="playerColorLabel" />
      <StatusItem label="Move" :value="moveCount" />
    </div>
  </BaseCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { BaseCard } from '@/components/atoms'
import StatusItem from './StatusItem.vue'
import type { Difficulty, Player } from '@/types/game'

interface Props {
  difficulty: Difficulty
  playerColor: Player
  moveCount: number
}

const props = defineProps<Props>()

const playerColorLabel = computed(() => {
  return props.playerColor === 1 ? 'Black' : 'White'
})
</script>

<style scoped>
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
</style>
