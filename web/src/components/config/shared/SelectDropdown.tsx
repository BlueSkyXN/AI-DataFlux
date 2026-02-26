/**
 * 下拉选择框组件
 *
 * 用途：渲染原生 select 下拉菜单，用于枚举值选择
 *
 * Props: value - 当前选中值 | options - 选项列表 | onChange - 选择回调
 */
interface SelectDropdownProps {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  disabled?: boolean;
}

/** 下拉选择框 */
export default function SelectDropdown({ value, options, onChange, disabled }: SelectDropdownProps) {
  return (
    <select
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent disabled:opacity-50 disabled:bg-gray-50 cursor-pointer"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
