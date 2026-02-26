/**
 * 原始 YAML 编辑器组件
 *
 * 用途：提供等宽字体文本域，允许用户直接编辑 YAML 配置原文
 *
 * 导出：RawYamlEditor（默认导出）
 *   Props: value       - 当前 YAML 文本
 *          onChange    - 文本变更回调
 *          loading     - 加载状态（加载中隐藏占位符）
 *          placeholder - 占位提示文本
 */
interface RawYamlEditorProps {
  value: string;
  onChange: (value: string) => void;
  loading?: boolean;
  placeholder?: string;
}

/** 原始 YAML 文本编辑器，使用等宽字体渲染 */
export default function RawYamlEditor({ value, onChange, loading, placeholder }: RawYamlEditorProps) {
  return (
    <div className="h-full flex flex-col bg-white rounded-2xl shadow-[0_2px_12px_rgba(0,0,0,0.04)] overflow-hidden border border-gray-100">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full flex-1 p-4 font-mono text-sm resize-none focus:outline-none bg-slate-50"
        style={{ fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', monospace", minHeight: '500px' }}
        placeholder={loading ? '' : placeholder}
        spellCheck={false}
      />
    </div>
  );
}
