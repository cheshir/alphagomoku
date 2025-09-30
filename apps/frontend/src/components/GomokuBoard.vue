<template>
  <div class="board-container">
    <svg
      :width="boardWidth"
      :height="boardHeight"
      class="gomoku-board"
      @click="handleBoardClick"
      @mousemove="handleMouseMove"
      @mouseleave="handleMouseLeave"
    >
      <!-- Wood texture background -->
      <defs>
        <pattern id="wood-texture" x="0" y="0" width="100%" height="100%">
          <rect width="100%" height="100%" :fill="woodColor" />
        </pattern>
        <radialGradient id="stone-gradient-black">
          <stop offset="0%" stop-color="#2a2a2a" />
          <stop offset="70%" stop-color="#0a0a0a" />
          <stop offset="100%" stop-color="#000000" />
        </radialGradient>
        <radialGradient id="stone-gradient-white">
          <stop offset="0%" stop-color="#ffffff" />
          <stop offset="70%" stop-color="#f0f0f0" />
          <stop offset="100%" stop-color="#d0d0d0" />
        </radialGradient>
      </defs>

      <!-- Board background -->
      <rect
        :width="boardWidth"
        :height="boardHeight"
        fill="url(#wood-texture)"
        class="board-bg"
      />

      <!-- Grid lines -->
      <g class="grid-lines">
        <!-- Horizontal lines -->
        <line
          v-for="i in boardSize"
          :key="`h-${i}`"
          :x1="padding"
          :y1="padding + (i - 1) * cellSize"
          :x2="boardWidth - padding"
          :y2="padding + (i - 1) * cellSize"
          stroke="#000000"
          stroke-width="1"
        />
        <!-- Vertical lines -->
        <line
          v-for="i in boardSize"
          :key="`v-${i}`"
          :x1="padding + (i - 1) * cellSize"
          :y1="padding"
          :x2="padding + (i - 1) * cellSize"
          :y2="boardHeight - padding"
          stroke="#000000"
          stroke-width="1"
        />
      </g>

      <!-- Star points (5 traditional dots for 15x15) -->
      <g class="star-points">
        <circle
          v-for="point in starPoints"
          :key="`star-${point.row}-${point.col}`"
          :cx="padding + point.col * cellSize"
          :cy="padding + point.row * cellSize"
          r="4"
          fill="#000000"
        />
      </g>

      <!-- Stones -->
      <g class="stones">
        <g
          v-for="(row, rowIndex) in board"
          :key="`row-${rowIndex}`"
        >
          <g
            v-for="(cell, colIndex) in row"
            :key="`cell-${rowIndex}-${colIndex}`"
          >
            <circle
              v-if="cell !== 0"
              :cx="padding + colIndex * cellSize"
              :cy="padding + rowIndex * cellSize"
              :r="stoneRadius"
              :fill="cell === 1 ? 'url(#stone-gradient-black)' : 'url(#stone-gradient-white)'"
              :class="[
                'stone',
                cell === 1 ? 'stone-black' : 'stone-white',
                isLastMove(rowIndex, colIndex) ? 'last-move' : ''
              ]"
            />
            <!-- Last move indicator -->
            <circle
              v-if="isLastMove(rowIndex, colIndex)"
              :cx="padding + colIndex * cellSize"
              :cy="padding + rowIndex * cellSize"
              :r="stoneRadius / 3"
              :fill="lastMove?.player === 1 ? '#ffffff' : '#000000'"
              class="last-move-marker"
            />
          </g>
        </g>
      </g>

      <!-- Hover indicator -->
      <circle
        v-if="hoverPosition && !isGameOver && isPlayerTurn && !isAiThinking"
        :cx="padding + hoverPosition.col * cellSize"
        :cy="padding + hoverPosition.row * cellSize"
        :r="stoneRadius"
        :fill="playerColor === 1 ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.5)'"
        class="hover-stone"
        pointer-events="none"
      />
    </svg>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Move } from '@/types/game'

interface Props {
  board: number[][]
  boardSize: number
  lastMove: Move | null
  isGameOver: boolean
  isPlayerTurn: boolean
  isAiThinking: boolean
  playerColor: number
}

const props = defineProps<Props>()
const emit = defineEmits<{
  makeMove: [row: number, col: number]
}>()

// Board dimensions
const cellSize = 40
const padding = 30
const stoneRadius = 16

const boardWidth = computed(() => padding * 2 + (props.boardSize - 1) * cellSize)
const boardHeight = computed(() => padding * 2 + (props.boardSize - 1) * cellSize)

// Modern wood color
const woodColor = '#d4a574'

// Star points for 15x15 board
const starPoints = [
  { row: 3, col: 3 },
  { row: 3, col: 11 },
  { row: 7, col: 7 },
  { row: 11, col: 3 },
  { row: 11, col: 11 }
]

const hoverPosition = ref<{ row: number; col: number } | null>(null)

function isLastMove(row: number, col: number): boolean {
  return props.lastMove?.row === row && props.lastMove?.col === col
}

function handleBoardClick(event: MouseEvent) {
  if (props.isGameOver || !props.isPlayerTurn || props.isAiThinking) return

  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  // Convert to board coordinates
  const col = Math.round((x - padding) / cellSize)
  const row = Math.round((y - padding) / cellSize)

  // Check if click is within bounds and cell is empty
  if (
    row >= 0 &&
    row < props.boardSize &&
    col >= 0 &&
    col < props.boardSize &&
    props.board[row][col] === 0
  ) {
    emit('makeMove', row, col)
  }
}

function handleMouseMove(event: MouseEvent) {
  if (props.isGameOver || !props.isPlayerTurn || props.isAiThinking) {
    hoverPosition.value = null
    return
  }

  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  const col = Math.round((x - padding) / cellSize)
  const row = Math.round((y - padding) / cellSize)

  if (
    row >= 0 &&
    row < props.boardSize &&
    col >= 0 &&
    col < props.boardSize &&
    props.board[row][col] === 0
  ) {
    hoverPosition.value = { row, col }
  } else {
    hoverPosition.value = null
  }
}

function handleMouseLeave() {
  hoverPosition.value = null
}
</script>

<script lang="ts">
export default {
  name: 'GomokuBoard'
}
</script>

<style scoped>
.board-container {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}

.gomoku-board {
  cursor: pointer;
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3));
  border-radius: 8px;
}

.board-bg {
  rx: 8px;
  ry: 8px;
}

.stone {
  transition: all 0.2s ease;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.4));
}

.stone-black {
  stroke: #000000;
  stroke-width: 0.5;
}

.stone-white {
  stroke: #888888;
  stroke-width: 0.5;
}

.last-move {
  animation: pulse 0.5s ease-in-out;
}

.last-move-marker {
  pointer-events: none;
}

.hover-stone {
  animation: fadeIn 0.2s ease;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>