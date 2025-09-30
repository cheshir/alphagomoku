<template>
  <div id="app">
    <header class="app-header">
      <h1>🎮 Gomoku AI</h1>
      <p class="subtitle">Play against AlphaZero-style AI</p>
    </header>

    <main class="app-main">
      <div class="game-layout">
        <aside class="sidebar">
          <GameControls
            :game-id="gameStore.gameId"
            :difficulty="gameStore.difficulty"
            :player-color="gameStore.playerColor"
            :move-count="gameStore.moveCount"
            :is-player-turn="gameStore.isPlayerTurn"
            :is-game-over="gameStore.isGameOver"
            :is-ai-thinking="gameStore.isAiThinking"
            :winner="gameStore.winner"
            @new-game="handleNewGame"
            @resign="handleResign"
            @reset-to-setup="handleResetToSetup"
          />

          <DebugPanel :debug-info="gameStore.debugInfo" />
        </aside>

        <section class="game-area">
          <GameTimer
            v-if="gameStore.gameId"
            :player-time="gameStore.playerTime"
            :ai-time="gameStore.aiTime"
            :player-color="gameStore.playerColor"
            :is-player-turn="gameStore.isPlayerTurn"
            :is-game-over="gameStore.isGameOver"
            :is-ai-thinking="gameStore.isAiThinking"
          />

          <GomokuBoard
            v-if="gameStore.board.length > 0"
            :board="gameStore.board"
            :board-size="gameStore.boardSize"
            :last-move="gameStore.lastMove"
            :is-game-over="gameStore.isGameOver"
            :is-player-turn="gameStore.isPlayerTurn"
            :is-ai-thinking="gameStore.isAiThinking"
            :player-color="gameStore.playerColor"
            @make-move="handleMakeMove"
          />

          <div v-else class="no-game">
            <p>Start a new game to begin playing!</p>
          </div>
        </section>
      </div>
    </main>

    <div v-if="gameStore.error" class="error-toast">
      {{ gameStore.error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { useGameStore } from '@/stores/game'
import GomokuBoard from '@/components/GomokuBoard.vue'
import GameControls from '@/components/GameControls.vue'
import GameTimer from '@/components/GameTimer.vue'
import DebugPanel from '@/components/DebugPanel.vue'
import type { Difficulty, Player } from '@/types/game'

const gameStore = useGameStore()

function handleResetToSetup() {
  gameStore.resetGame()
}

async function handleNewGame(difficulty: Difficulty, color: Player) {
  try {
    gameStore.resetGame()
    await gameStore.createGame(difficulty, color)
  } catch (error) {
    console.error('Failed to create game:', error)
  }
}

async function handleMakeMove(row: number, col: number) {
  try {
    await gameStore.makeMove(row, col)
  } catch (error) {
    console.error('Failed to make move:', error)
  }
}

async function handleResign() {
  try {
    await gameStore.resign()
  } catch (error) {
    console.error('Failed to resign:', error)
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
    sans-serif;
  background: linear-gradient(135deg, #04112c 0%, #49356a 100%);
  min-height: 100vh;
}

#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  text-align: center;
  padding: 32px 20px;
  color: white;
}

.app-header h1 {
  font-size: 48px;
  font-weight: 700;
  margin-bottom: 8px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
  font-size: 18px;
  opacity: 0.9;
}

.app-main {
  flex: 1;
  padding: 20px;
  max-width: 1600px;
  width: 100%;
  margin: 0 auto;
}

.game-layout {
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 24px;
  align-items: start;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: sticky;
  top: 20px;
}

.game-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 32px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  min-height: 600px;
}

.no-game {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: #999;
  font-size: 18px;
  font-style: italic;
}

.error-toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: #e74c3c;
  color: white;
  padding: 16px 24px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  animation: slideIn 0.3s ease;
  max-width: 400px;
}

@keyframes slideIn {
  from {
    transform: translateX(400px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@media (max-width: 1200px) {
  .game-layout {
    grid-template-columns: 1fr;
  }

  .sidebar {
    position: relative;
    top: 0;
  }
}
</style>