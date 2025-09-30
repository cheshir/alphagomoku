<template>
  <div class="status-item">
    <span class="status-item__label">{{ label }}:</span>
    <span class="status-item__value">
      <slot>{{ value }}</slot>
    </span>
  </div>
</template>

<script setup lang="ts">
interface Props {
  label: string
  value?: string | number
}

withDefaults(defineProps<Props>(), {
  value: undefined
})
</script>

<style scoped>
.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.status-item:last-child {
  border-bottom: none;
}

.status-item__label {
  font-weight: 500;
  color: #666;
  font-size: 14px;
}

.status-item__value {
  font-weight: 600;
  color: #333;
  font-size: 14px;
}
</style>
