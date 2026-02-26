/**
 * 表单字段包装组件
 *
 * 用途：统一渲染表单标签、必填标记、描述文本，支持垂直和水平布局
 *
 * Props: label - 字段标签 | description - 描述文本 | required - 是否必填
 *        horizontal - 是否水平布局 | children - 表单控件
 */
interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  children: React.ReactNode;
  horizontal?: boolean;
}

/** 表单字段容器，支持垂直（默认）和水平两种布局 */
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
