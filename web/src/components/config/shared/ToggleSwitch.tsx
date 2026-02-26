/**
 * 开关切换组件
 *
 * 用途：布尔值的可视化切换控件，遵循 ARIA switch 角色规范
 *
 * Props: checked - 当前状态 | onChange - 状态变更回调 | disabled - 是否禁用
 */
interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

/** 开关切换按钮，带滑动动画效果 */
export default function ToggleSwitch({ checked, onChange, disabled }: ToggleSwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed ${
        checked ? 'bg-cyan-500' : 'bg-gray-200'
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow-lg ring-0 transition-transform duration-200 ease-in-out ${
          checked ? 'translate-x-5' : 'translate-x-0'
        }`}
      />
    </button>
  );
}
