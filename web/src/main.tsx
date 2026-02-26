/**
 * 应用入口文件
 *
 * 职责：挂载 React 根组件到 DOM，启用 StrictMode 严格模式。
 *
 * 依赖模块：
 * - react / react-dom — React 核心渲染库
 * - index.css — 全局样式（Tailwind CSS）
 * - App — 主应用组件
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// 获取 DOM 根节点并渲染 React 应用
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
