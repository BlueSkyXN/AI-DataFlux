/**
 * 键值对编辑器组件
 *
 * 用途：管理字符串键值对的增删改操作，适用于列名映射等场景
 *
 * Props: value - 键值对对象 | onChange - 变更回调
 *        keyPlaceholder/valuePlaceholder - 输入占位文本 | addLabel - 添加按钮文本
 */
import { useState } from 'react';
interface KeyValueEditorProps {
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
  addLabel?: string;
}

/** 键值对编辑器，支持新增、删除、编辑值 */
export default function KeyValueEditor({
  value = {},
  onChange,
  keyPlaceholder = 'key',
  valuePlaceholder = 'value',
  addLabel = '+',
}: KeyValueEditorProps) {
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');

  const entries = Object.entries(value);

  const handleAdd = () => {
    const k = newKey.trim();
    const v = newValue.trim();
    if (!k) return;
    onChange({ ...value, [k]: v });
    setNewKey('');
    setNewValue('');
  };

  const handleRemove = (key: string) => {
    const next = { ...value };
    delete next[key];
    onChange(next);
  };

  const handleValueChange = (key: string, newVal: string) => {
    onChange({ ...value, [key]: newVal });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAdd();
    }
  };

  return (
    <div className="space-y-2">
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center gap-2">
          <span className="px-2.5 py-1.5 bg-gray-100 text-gray-700 text-sm rounded-lg font-mono min-w-[80px]">
            {k}
          </span>
          <svg className="w-4 h-4 text-gray-300 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
          <input
            type="text"
            value={v}
            onChange={(e) => handleValueChange(k, e.target.value)}
            className="flex-1 px-2.5 py-1.5 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
          />
          <button
            type="button"
            onClick={() => handleRemove(k)}
            className="text-gray-400 hover:text-red-500 p-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={keyPlaceholder}
          className="w-[120px] px-2.5 py-1.5 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
        />
        <svg className="w-4 h-4 text-gray-300 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
        </svg>
        <input
          type="text"
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={valuePlaceholder}
          className="flex-1 px-2.5 py-1.5 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
        />
        <button
          type="button"
          onClick={handleAdd}
          disabled={!newKey.trim()}
          className="px-3 py-1.5 text-sm font-medium text-cyan-600 bg-cyan-50 rounded-lg hover:bg-cyan-100 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {addLabel}
        </button>
      </div>
    </div>
  );
}
