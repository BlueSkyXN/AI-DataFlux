/**
 * 数字输入框组件
 *
 * 用途：带范围限制和步长控制的数字输入，自动根据步长选择整数/浮点解析
 *
 * Props: value - 当前值 | onChange - 值变更回调 | min/max/step - 范围和步长
 */
interface NumberInputProps {
  value: number | undefined;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
}

/** 数字输入框，step < 1 时使用 parseFloat，否则使用 parseInt */
export default function NumberInput({ value, onChange, min, max, step, placeholder, disabled }: NumberInputProps) {
  return (
    <input
      type="number"
      value={value ?? ''}
      onChange={(e) => {
        const v = e.target.value;
        if (v === '') return;
        const num = step && step < 1 ? parseFloat(v) : parseInt(v, 10);
        if (!isNaN(num)) onChange(num);
      }}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      disabled={disabled}
      className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent disabled:opacity-50 disabled:bg-gray-50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
    />
  );
}
