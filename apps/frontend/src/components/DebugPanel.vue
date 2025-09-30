<template>
  <div class="debug-panel">
    <div class="debug-header" @click="toggleExpanded">
      <h3>Debug Information</h3>
      <span class="toggle-icon">{{ isExpanded ? '▼' : '▶' }}</span>
    </div>

    <div v-if="isExpanded && debugInfo" class="debug-content">
      <div class="debug-section">
        <h4>Search Metrics</h4>
        <div class="metric-grid">
          <div class="metric-item">
            <span class="metric-label">Simulations:</span>
            <span class="metric-value">{{ debugInfo.simulations }}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Thinking Time:</span>
            <span class="metric-value">{{ debugInfo.thinking_time_ms.toFixed(1) }}ms</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Search Depth:</span>
            <span class="metric-value">{{ debugInfo.search_depth }}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Nodes Explored:</span>
            <span class="metric-value">{{ debugInfo.nodes_explored }}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Value Estimate:</span>
            <span class="metric-value" :class="valueClass">
              {{ (debugInfo.value_estimate * 100).toFixed(1) }}%
            </span>
          </div>
        </div>
      </div>

      <div class="debug-section">
        <h4>Top Moves</h4>
        <div class="top-moves-list">
          <div
            v-for="(move, index) in debugInfo.top_moves"
            :key="`move-${index}`"
            class="top-move-item"
          >
            <span class="move-rank">{{ index + 1 }}.</span>
            <span class="move-position">
              ({{ move.row }}, {{ move.col }})
            </span>
            <div class="move-bar-container">
              <div
                class="move-bar"
                :style="{ width: `${move.probability * 100}%` }"
              ></div>
            </div>
            <span class="move-probability">
              {{ (move.probability * 100).toFixed(1) }}%
            </span>
            <span class="move-visits">
              {{ move.visits }} visits
            </span>
          </div>
        </div>
      </div>

      <div v-if="showPolicyDistribution" class="debug-section">
        <div class="section-header">
          <h4>Policy Distribution</h4>
          <button @click="togglePolicy" class="toggle-btn">
            {{ showPolicyMap ? 'Hide' : 'Show' }} Heatmap
          </button>
        </div>
        <div v-if="showPolicyMap && debugInfo.policy_distribution" class="policy-heatmap">
          <div class="heatmap-grid">
            <div
              v-for="(row, rowIndex) in debugInfo.policy_distribution"
              :key="`policy-row-${rowIndex}`"
              class="heatmap-row"
            >
              <div
                v-for="(prob, colIndex) in row"
                :key="`policy-cell-${rowIndex}-${colIndex}`"
                class="heatmap-cell"
                :style="{ background: getHeatmapColor(prob) }"
                :title="`(${rowIndex}, ${colIndex}): ${(prob * 100).toFixed(2)}%`"
              ></div>
            </div>
          </div>
          <div class="heatmap-legend">
            <span>Low probability</span>
            <div class="legend-gradient"></div>
            <span>High probability</span>
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="isExpanded && !debugInfo" class="no-data">
      No debug information available. Make a move to see AI metrics.
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { DebugInfo } from '@/types/game'

interface Props {
  debugInfo: DebugInfo | null
}

const props = defineProps<Props>()

const isExpanded = ref(false)
const showPolicyMap = ref(false)

const valueClass = computed(() => {
  if (!props.debugInfo) return ''
  const value = props.debugInfo.value_estimate
  if (value > 0.2) return 'positive'
  if (value < -0.2) return 'negative'
  return 'neutral'
})

const showPolicyDistribution = computed(() => {
  return props.debugInfo?.policy_distribution !== null
})

function toggleExpanded() {
  isExpanded.value = !isExpanded.value
}

function togglePolicy() {
  showPolicyMap.value = !showPolicyMap.value
}

function getHeatmapColor(probability: number): string {
  // Convert probability to color (blue to red gradient)
  const maxProb = 0.5 // Cap for better visualization
  const normalized = Math.min(probability / maxProb, 1)

  if (normalized === 0) return '#f0f0f0'

  const r = Math.floor(normalized * 255)
  const g = Math.floor((1 - normalized) * 180)
  const b = Math.floor((1 - normalized) * 255)

  return `rgb(${r}, ${g}, ${b})`
}
</script>

<style scoped>
.debug-panel {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  margin-top: 20px;
}

.debug-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: #f8f9fa;
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;
}

.debug-header:hover {
  background: #e9ecef;
}

.debug-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.toggle-icon {
  font-size: 14px;
  color: #666;
}

.debug-content {
  padding: 20px;
}

.debug-section {
  margin-bottom: 24px;
}

.debug-section:last-child {
  margin-bottom: 0;
}

.debug-section h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.toggle-btn {
  padding: 6px 12px;
  background: #4a90e2;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.2s;
}

.toggle-btn:hover {
  background: #357abd;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
}

.metric-label {
  color: #666;
  font-size: 14px;
  font-weight: 500;
}

.metric-value {
  font-weight: 700;
  font-size: 14px;
  color: #333;
}

.metric-value.positive {
  color: #27ae60;
}

.metric-value.negative {
  color: #e74c3c;
}

.metric-value.neutral {
  color: #f39c12;
}

.top-moves-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.top-move-item {
  display: grid;
  grid-template-columns: 25px 80px 1fr 60px 80px;
  align-items: center;
  gap: 12px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 6px;
  font-size: 13px;
}

.move-rank {
  font-weight: 700;
  color: #666;
}

.move-position {
  font-family: 'Courier New', monospace;
  color: #333;
  font-weight: 600;
}

.move-bar-container {
  height: 20px;
  background: #e0e0e0;
  border-radius: 10px;
  overflow: hidden;
}

.move-bar {
  height: 100%;
  background: linear-gradient(90deg, #4a90e2 0%, #357abd 100%);
  transition: width 0.3s ease;
}

.move-probability {
  font-weight: 600;
  color: #4a90e2;
  text-align: right;
}

.move-visits {
  color: #666;
  font-size: 12px;
  text-align: right;
}

.policy-heatmap {
  margin-top: 12px;
}

.heatmap-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
  overflow-x: auto;
}

.heatmap-row {
  display: flex;
  gap: 2px;
}

.heatmap-cell {
  width: 20px;
  height: 20px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: transform 0.1s;
}

.heatmap-cell:hover {
  transform: scale(1.2);
  z-index: 10;
  border-color: #333;
}

.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
  font-size: 12px;
  color: #666;
}

.legend-gradient {
  flex: 1;
  height: 20px;
  background: linear-gradient(90deg, #f0f0f0 0%, rgb(255, 0, 0) 100%);
  border-radius: 4px;
  border: 1px solid #ddd;
}

.no-data {
  padding: 40px 20px;
  text-align: center;
  color: #999;
  font-style: italic;
}
</style>