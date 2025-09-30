<template>
  <div class="form-field">
    <BaseLabel :html-for="id" :required="required">
      {{ label }}
    </BaseLabel>
    <slot />
    <span v-if="error" class="error-message">{{ error }}</span>
    <span v-if="hint && !error" class="hint-message">{{ hint }}</span>
  </div>
</template>

<script setup lang="ts">
import { BaseLabel } from '@/components/atoms'

interface Props {
  label: string
  id?: string
  required?: boolean
  error?: string
  hint?: string
}

withDefaults(defineProps<Props>(), {
  id: undefined,
  required: false,
  error: undefined,
  hint: undefined
})
</script>

<style scoped>
.form-field {
  margin-bottom: 20px;
}

.error-message {
  display: block;
  margin-top: 6px;
  font-size: 13px;
  color: #dc3545;
}

.hint-message {
  display: block;
  margin-top: 6px;
  font-size: 13px;
  color: #6c757d;
}
</style>
