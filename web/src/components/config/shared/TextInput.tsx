interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'url' | 'password';
  monospace?: boolean;
  disabled?: boolean;
}

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
