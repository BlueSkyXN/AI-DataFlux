/**
 * Vite 构建配置 — AI-DataFlux Web GUI 前端
 *
 * 构建命令: npm run build → tsc -b && vite build → web/dist/
 * 开发命令: npm run dev → 启动开发服务器（含 HMR 热更新）
 *
 * 插件:
 *   - @vitejs/plugin-react: React JSX 转换和 Fast Refresh
 *   - @tailwindcss/vite: Tailwind CSS 4 集成
 *
 * 代理配置:
 *   - /api → 控制面板后端 (默认 http://127.0.0.1:8790)
 *   - 支持 WebSocket 代理（用于实时日志流）
 */
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // 开发环境下将 /api 请求代理到控制面板后端服务
      '/api': {
        target: process.env.VITE_CONTROL_SERVER || 'http://127.0.0.1:8790',
        changeOrigin: true,  // 修改请求 Host 头为目标地址
        ws: true,            // 启用 WebSocket 代理（用于实时日志推送）
      },
    },
  },
})
