interface NumberInputProps {
  value: number | undefined;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
}

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
