/**
 * 文本输入框组件
 *
 * 用途：通用单行文本输入，支持 text/url/password 类型和等宽字体模式
 *
 * Props: value - 文本值 | onChange - 变更回调 | type - 输入类型
 *        monospace - 是否等宽字体
 */
interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'url' | 'password';
  monospace?: boolean;
  disabled?: boolean;
}

/** 单行文本输入框 */
export default function TextInput({ value, onChange, placeholder, type = 'text', monospace, disabled }: TextInputProps) {
  return (
    <input
      type={type}
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className={`w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent disabled:opacity-50 disabled:bg-gray-50 ${
        monospace ? "font-mono" : ""
      }`}
    />
  );
}
