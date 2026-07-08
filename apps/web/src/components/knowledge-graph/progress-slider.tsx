'use client';
export function ProgressSlider() {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-600">学习进度模拟</label>
      <input
        type="range"
        min="0"
        max="100"
        defaultValue={0}
        className="w-full accent-primary"
      />
      <div className="flex justify-between text-xs text-gray-400">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
