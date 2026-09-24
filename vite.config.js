import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // only scan/watch the app itself (agent worktrees live under .claude/)
  optimizeDeps: { entries: ['index.html'] },
  server: { port: 5173, strictPort: false, watch: { ignored: ['**/.claude/**', '**/captures/**', '**/.qa/**'] } },
  build: { target: 'es2022', assetsInlineLimit: 0, chunkSizeWarningLimit: 2000 },
});
