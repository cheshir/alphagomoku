<template>
  <label
    :class="['base-label', { 'base-label--required': required }]"
    :for="htmlFor"
  >
    <slot />
    <span v-if="required" class="required-marker">*</span>
  </label>
</template>

<script setup lang="ts">
interface Props {
  htmlFor?: string
  required?: boolean
}

withDefaults(defineProps<Props>(), {
  htmlFor: undefined,
  required: false
})
</script>

<style scoped>
.base-label {
  display: block;
  font-weight: 600;
  color: #333;
  margin-bottom: 6px;
  font-size: 14px;
}

.required-marker {
  color: #dc3545;
  margin-left: 4px;
}
</style>
