/**
 * 配置区块卡片组件
 *
 * 用途：作为各配置分区的外层容器，提供统一的标题、描述和内容区域样式
 *
 * Props: title - 卡片标题 | description - 描述文本 | children - 内容区域
 */
interface SectionCardProps {
  title: string;
  description?: string;
  children: React.ReactNode;
}

/** 配置区块卡片容器 */
export default function SectionCard({ title, description, children }: SectionCardProps) {
  return (
    <div className="bg-white rounded-2xl p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
      <h3 className="text-base font-semibold text-gray-800 mb-1">{title}</h3>
      {description && (
        <p className="text-sm text-gray-500 mb-4">{description}</p>
      )}
      {!description && <div className="mb-4" />}
      <div className="space-y-4">{children}</div>
    </div>
  );
}
