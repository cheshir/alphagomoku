<template>
  <div class="game-actions">
    <BaseButton
      v-if="showResign"
      variant="danger"
      :disabled="disabled"
      @click="handleResign"
    >
      Resign
    </BaseButton>
    <BaseButton
      variant="primary"
      @click="$emit('newGame')"
    >
      New Game
    </BaseButton>
  </div>
</template>

<script setup lang="ts">
import { BaseButton } from '@/components/atoms'

interface Props {
  showResign?: boolean
  disabled?: boolean
}

withDefaults(defineProps<Props>(), {
  showResign: true,
  disabled: false
})

const emit = defineEmits<{
  resign: []
  newGame: []
}>()

function handleResign() {
  if (confirm('Are you sure you want to resign?')) {
    emit('resign')
  }
}
</script>

<style scoped>
.game-actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
</style>
