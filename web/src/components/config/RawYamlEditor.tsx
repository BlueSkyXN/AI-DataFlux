interface RawYamlEditorProps {
  value: string;
  onChange: (value: string) => void;
  loading?: boolean;
  placeholder?: string;
}

export default function RawYamlEditor({ value, onChange, loading, placeholder }: RawYamlEditorProps) {
  return (
    <div className="h-full bg-white rounded-2xl shadow-[0_2px_12px_rgba(0,0,0,0.04)] overflow-hidden">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-full p-4 font-mono text-sm resize-none focus:outline-none bg-slate-50"
        style={{ fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', monospace" }}
        placeholder={loading ? '' : placeholder}
        spellCheck={false}
      />
    </div>
  );
}
