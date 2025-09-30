<template>
  <div :class="['base-spinner', `base-spinner--${size}`]"></div>
</template>

<script setup lang="ts">
interface Props {
  size?: 'small' | 'medium' | 'large'
}

withDefaults(defineProps<Props>(), {
  size: 'medium'
})
</script>

<style scoped>
.base-spinner {
  border-radius: 50%;
  border-style: solid;
  border-color: currentColor transparent transparent transparent;
  animation: spin 1s linear infinite;
}

.base-spinner--small {
  width: 16px;
  height: 16px;
  border-width: 2px;
}

.base-spinner--medium {
  width: 24px;
  height: 24px;
  border-width: 3px;
}

.base-spinner--large {
  width: 40px;
  height: 40px;
  border-width: 4px;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>
