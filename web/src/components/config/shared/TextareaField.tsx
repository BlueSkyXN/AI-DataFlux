interface TextareaFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  monospace?: boolean;
  disabled?: boolean;
}

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
