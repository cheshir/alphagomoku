<template>
  <select
    :class="['base-select', { 'base-select--error': error }]"
    :value="modelValue"
    @change="handleChange"
  >
    <option
      v-for="option in options"
      :key="option.value"
      :value="option.value"
    >
      {{ option.label }}
    </option>
  </select>
</template>

<script setup lang="ts">
interface Option {
  value: string | number
  label: string
}

interface Props {
  modelValue: string | number
  options: Option[]
  error?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  error: false
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number]
}>()

function handleChange(event: Event) {
  const target = event.target as HTMLSelectElement
  const value = target.value
  // Try to parse as number if the original modelValue is a number
  const parsedValue = !isNaN(Number(value)) && typeof props.modelValue === 'number'
    ? Number(value)
    : value
  emit('update:modelValue', parsedValue)
}
</script>

<style scoped>
.base-select {
  width: 100%;
  padding: 10px 12px;
  font-size: 16px;
  font-family: inherit;
  border: 2px solid #ddd;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  transition: all 0.2s ease;
  outline: none;
}

.base-select:hover {
  border-color: #bbb;
}

.base-select:focus {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.base-select--error {
  border-color: #dc3545;
}

.base-select--error:focus {
  border-color: #dc3545;
  box-shadow: 0 0 0 3px rgba(220, 53, 69, 0.1);
}
</style>
