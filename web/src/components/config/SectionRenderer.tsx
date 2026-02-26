/**
 * 配置分区路由渲染器
 *
 * 用途：根据当前选中的配置分区 ID，动态渲染对应的配置表单组件
 *
 * 导出：
 *   - SectionProps（接口）：各分区组件的通用 Props 定义
 *   - SectionRenderer（默认导出）：分区路由组件
 *
 * 依赖：./ConfigSidebar（ConfigSectionId 类型）、./sections/*（各分区组件）
 */
import type { ConfigSectionId } from './ConfigSidebar';
import type { Language } from '../../i18n';
import GlobalSection from './sections/GlobalSection';
import DatasourceSection from './sections/DatasourceSection';
import ConcurrencySection from './sections/ConcurrencySection';
import ColumnsSection from './sections/ColumnsSection';
import ValidationSection from './sections/ValidationSection';
import ModelsSection from './sections/ModelsSection';
import ChannelsSection from './sections/ChannelsSection';
import PromptSection from './sections/PromptSection';
import TokenSection from './sections/TokenSection';
import RoutingSection from './sections/RoutingSection';

/** 各配置分区组件共享的 Props 接口 */
export interface SectionProps {
  /** 完整的表单数据对象 */
  formData: Record<string, unknown>;
  /** 按路径更新配置值 */
  updateConfig: (path: string[], value: unknown) => void;
  /** 按路径读取配置值 */
  getConfig: (path: string[]) => unknown;
  /** 当前界面语言 */
  language: Language;
}

/** SectionRenderer 的 Props，在 SectionProps 基础上增加当前分区标识 */
interface SectionRendererProps extends SectionProps {
  section: ConfigSectionId;
}

/**
 * 分区路由渲染器，根据 section ID 渲染对应的配置表单组件
 * @returns 匹配的分区组件，未匹配时返回 null
 */
export default function SectionRenderer({ section, ...props }: SectionRendererProps) {
  switch (section) {
    case 'global':
      return <GlobalSection {...props} />;
    case 'datasource':
      return <DatasourceSection {...props} />;
    case 'concurrency':
      return <ConcurrencySection {...props} />;
    case 'columns':
      return <ColumnsSection {...props} />;
    case 'validation':
      return <ValidationSection {...props} />;
    case 'models':
      return <ModelsSection {...props} />;
    case 'channels':
      return <ChannelsSection {...props} />;
    case 'prompt':
      return <PromptSection {...props} />;
    case 'token':
      return <TokenSection {...props} />;
    case 'routing':
      return <RoutingSection {...props} />;
    default:
      return null;
  }
}
