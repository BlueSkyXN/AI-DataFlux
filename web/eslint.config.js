/**
 * ESLint 配置文件 — AI-DataFlux Web GUI 前端
 *
 * 技术栈: React 19 + TypeScript 5.9
 * 规则集: ESLint 推荐 + TypeScript ESLint 推荐 + React Hooks + React Refresh (Vite HMR)
 * 检查范围: 所有 .ts/.tsx 文件，排除 dist 构建输出目录
 */
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  // 忽略构建输出目录
  globalIgnores(['dist']),
  {
    // 仅对 TypeScript 文件生效
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,             // ESLint 基础推荐规则
      tseslint.configs.recommended,        // TypeScript ESLint 推荐规则
      reactHooks.configs.flat.recommended, // React Hooks 规则（依赖数组检查等）
      reactRefresh.configs.vite,           // React Refresh 规则（确保 Vite HMR 正常工作）
    ],
    languageOptions: {
      ecmaVersion: 2020,        // ECMAScript 2020 语法支持
      globals: globals.browser,  // 浏览器全局变量（window, document 等）
    },
  },
])
