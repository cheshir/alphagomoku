<template>
  <div :class="['base-card', { 'base-card--elevated': elevated }]">
    <slot />
  </div>
</template>

<script setup lang="ts">
interface Props {
  elevated?: boolean
}

withDefaults(defineProps<Props>(), {
  elevated: false
})
</script>

<style scoped>
.base-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: box-shadow 0.2s ease;
}

.base-card--elevated {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.base-card--elevated:hover {
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}
</style>
