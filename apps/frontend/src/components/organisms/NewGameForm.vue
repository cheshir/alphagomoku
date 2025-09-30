<template>
  <div class="new-game-form">
    <h2>New Game</h2>

    <FormField
      label="Difficulty"
      id="difficulty"
    >
      <BaseSelect
        v-model="selectedDifficulty"
        :options="difficultyOptions"
      />
    </FormField>

    <FormField
      label="Play as"
      id="color"
    >
      <BaseSelect
        v-model="selectedColor"
        :options="colorOptions"
      />
    </FormField>

    <BaseButton
      variant="primary"
      size="large"
      @click="handleStartGame"
    >
      Start Game
    </BaseButton>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { BaseButton, BaseSelect } from '@/components/atoms'
import { FormField } from '@/components/molecules'
import type { Difficulty, Player } from '@/types/game'

const emit = defineEmits<{
  startGame: [difficulty: Difficulty, color: Player]
}>()

const selectedDifficulty = ref<Difficulty>('medium')
const selectedColor = ref<Player>(1)

const difficultyOptions = [
  { value: 'easy', label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard', label: 'Hard' }
]

const colorOptions = [
  { value: 1, label: 'Black (First)' },
  { value: -1, label: 'White (Second)' }
]

function handleStartGame() {
  emit('startGame', selectedDifficulty.value, selectedColor.value)
}
</script>

<style scoped>
.new-game-form {
  padding: 24px;
}

.new-game-form h2 {
  margin: 0 0 20px 0;
  font-size: 24px;
  font-weight: 600;
  color: #333;
}
</style>
