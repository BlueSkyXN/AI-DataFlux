/**
 * 可折叠数组项卡片组件
 *
 * 用途：以卡片形式展示数组中的单个条目（如模型、渠道），
 *       支持折叠/展开、删除、复制操作
 *
 * Props: title - 卡片标题 | subtitle - 副标题 | onRemove - 删除回调
 *        onDuplicate - 复制回调（可选）| defaultCollapsed - 默认折叠状态
 */
import { useState } from 'react';
interface ArrayItemCardProps {
  title: string;
  subtitle?: string;
  onRemove: () => void;
  onDuplicate?: () => void;
  defaultCollapsed?: boolean;
  children: React.ReactNode;
}

/** 可折叠数组项卡片，点击头部切换展开/收起 */
export default function ArrayItemCard({ title, subtitle, onRemove, onDuplicate, defaultCollapsed = true, children }: ArrayItemCardProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      {/* Header */}
      {/* 卡片头部：标题 + 操作按钮 */}
      <div
        className="flex items-center gap-3 px-4 py-3 bg-gray-50 cursor-pointer select-none"
        onClick={() => setCollapsed(!collapsed)}
      >
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${collapsed ? '' : 'rotate-90'}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-sm font-medium text-gray-800">{title}</span>
        {subtitle && (
          <span className="text-xs text-gray-400">{subtitle}</span>
        )}
        <div className="flex-1" />
        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
          {onDuplicate && (
            <button
              type="button"
              onClick={onDuplicate}
              className="p-1.5 text-gray-400 hover:text-cyan-500 rounded"
              title="Duplicate"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
          )}
          <button
            type="button"
            onClick={onRemove}
            className="p-1.5 text-gray-400 hover:text-red-500 rounded"
            title="Remove"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>
      {/* Body */}
      {/* 卡片内容区：仅在展开时渲染 */}
      {!collapsed && (
        <div className="p-4 space-y-4">
          {children}
        </div>
      )}
    </div>
  );
}
