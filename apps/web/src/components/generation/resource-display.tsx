'use client';

import { ResourcePreview } from '@/stores/generation-store';

interface ResourcePreviewProps {
  resources: ResourcePreview[];
}

const RESOURCE_LABELS: Record<string, string> = {
  explanation: '讲解',
  exercise: '练习',
  visualization: '可视化',
  code: '代码',
};

const RESOURCE_COLORS: Record<string, string> = {
  explanation: 'border-blue-200 bg-blue-50',
  exercise: 'border-green-200 bg-green-50',
  visualization: 'border-purple-200 bg-purple-50',
  code: 'border-amber-200 bg-amber-50',
};

export function ResourcePreviewList({ resources }: ResourcePreviewProps) {
  if (!resources.length) {
    return null;
  }

  const grouped = resources.reduce<Record<string, ResourcePreview[]>>((acc, r) => {
    const key = r.type || 'other';
    if (!acc[key]) acc[key] = [];
    acc[key].push(r);
    return acc;
  }, {});

  return (
    <div className="rounded-xl border bg-white p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        生成结果 ({resources.length} 项)
      </h2>

      <div className="space-y-3">
        {Object.entries(grouped).map(([type, items]) => (
          <div key={type} className={`rounded-lg border p-3 ${RESOURCE_COLORS[type] ?? 'border-gray-200 bg-gray-50'}`}>
            <span className="inline-block rounded bg-white px-2 py-0.5 text-xs font-medium text-gray-600 mb-2">
              {RESOURCE_LABELS[type] ?? type}
            </span>
            <ul className="space-y-1">
              {items.map((item, i) => (
                <li key={i} className="text-sm text-gray-700 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-current shrink-0" />
                  {item.title}
                  {item.kp_id ? (
                    <span className="text-xs text-gray-400">({item.kp_id})</span>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
