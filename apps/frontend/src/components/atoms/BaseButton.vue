<template>
  <button
    :class="['base-button', `base-button--${variant}`, `base-button--${size}`, { 'base-button--disabled': disabled }]"
    :disabled="disabled"
    @click="$emit('click', $event)"
  >
    <slot />
  </button>
</template>

<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'secondary' | 'danger'
  size?: 'small' | 'medium' | 'large'
  disabled?: boolean
}

withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'medium',
  disabled: false
})

defineEmits<{
  click: [event: MouseEvent]
}>()
</script>

<style scoped>
.base-button {
  font-family: inherit;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  outline: none;
}

.base-button:focus-visible {
  outline: 2px solid #4a90e2;
  outline-offset: 2px;
}

/* Sizes */
.base-button--small {
  padding: 8px 16px;
  font-size: 14px;
}

.base-button--medium {
  padding: 12px 24px;
  font-size: 16px;
}

.base-button--large {
  padding: 16px 32px;
  font-size: 18px;
}

/* Variants */
.base-button--primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.base-button--primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.base-button--primary:active:not(:disabled) {
  transform: translateY(0);
}

.base-button--secondary {
  background: #6c757d;
  color: white;
}

.base-button--secondary:hover:not(:disabled) {
  background: #5a6268;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(108, 117, 125, 0.3);
}

.base-button--danger {
  background: #dc3545;
  color: white;
}

.base-button--danger:hover:not(:disabled) {
  background: #c82333;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3);
}

.base-button--disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
