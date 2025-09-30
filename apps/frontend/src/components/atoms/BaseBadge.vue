<template>
  <span :class="['base-badge', `base-badge--${variant}`]">
    <slot />
  </span>
</template>

<script setup lang="ts">
interface Props {
  variant?: 'success' | 'warning' | 'danger' | 'info' | 'neutral'
}

withDefaults(defineProps<Props>(), {
  variant: 'neutral'
})
</script>

<style scoped>
.base-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.base-badge--success {
  background: #d4edda;
  color: #155724;
}

.base-badge--warning {
  background: #fff3cd;
  color: #856404;
}

.base-badge--danger {
  background: #f8d7da;
  color: #721c24;
}

.base-badge--info {
  background: #d1ecf1;
  color: #0c5460;
}

.base-badge--neutral {
  background: #e9ecef;
  color: #495057;
}
</style>
