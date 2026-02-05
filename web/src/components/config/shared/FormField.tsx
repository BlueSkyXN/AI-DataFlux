interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  children: React.ReactNode;
  horizontal?: boolean;
}

export default function FormField({ label, description, required, children, horizontal }: FormFieldProps) {
  if (horizontal) {
    return (
      <div className="flex items-center justify-between gap-4">
        <div className="shrink-0">
          <span className="text-sm font-medium text-gray-700">
            {label}
            {required && <span className="text-red-400 ml-0.5">*</span>}
          </span>
          {description && (
            <p className="text-xs text-gray-400 mt-0.5">{description}</p>
          )}
        </div>
        <div className="shrink-0">{children}</div>
      </div>
    );
  }

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-400 ml-0.5">*</span>}
      </label>
      {description && (
        <p className="text-xs text-gray-400 mb-1.5">{description}</p>
      )}
      {children}
    </div>
  );
}
