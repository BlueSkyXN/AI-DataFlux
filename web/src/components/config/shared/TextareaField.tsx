/**
 * 多行文本输入框组件
 *
 * 用途：用于系统提示词、模板等长文本内容的编辑
 *
 * Props: value - 文本内容 | onChange - 变更回调 | rows - 行数
 *        monospace - 是否等宽字体 | placeholder - 占位文本
 */
interface TextareaFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  monospace?: boolean;
  disabled?: boolean;
}

/** 多行文本输入框 */
export default function TextareaField({ value, onChange, placeholder, rows = 6, monospace = true, disabled }: TextareaFieldProps) {
  return (
    <textarea
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      rows={rows}
      disabled={disabled}
      spellCheck={false}
      className={`w-full px-3 py-2 text-sm border border-gray-200 rounded-lg resize-y focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent disabled:opacity-50 disabled:bg-gray-50 ${
        monospace ? "font-mono" : ""
      }`}
    />
  );
}
