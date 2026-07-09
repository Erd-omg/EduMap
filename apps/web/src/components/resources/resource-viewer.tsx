'use client';

interface ResourceViewerProps {
  kpId?: string;
  kpName?: string;
  contentType?: string;
  content?: string;
  onClose?: () => void;
}

export function ResourceViewer({
  kpId,
  kpName,
  contentType,
  content,
  onClose,
}: ResourceViewerProps) {
  if (!kpId) {
    return (
      <div className="rounded-xl border bg-white p-6">
        <p className="text-center text-sm text-gray-400">
          选择技能树中的知识点查看学习资源
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border bg-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{kpName || kpId}</h3>
          {contentType && (
            <span className="mt-1 inline-block rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-600">
              {contentType}
            </span>
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-sm text-gray-400 hover:text-gray-600"
          >
            关闭
          </button>
        )}
      </div>

      {/* Content */}
      {content ? (
        <div className="prose prose-sm max-w-none text-gray-700">
          {content.split('\n').map((line, i) => (
            <p key={i} className="mb-2">
              {line}
            </p>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-400 py-4 text-center">
          该知识点暂无生成资源，请使用"生成资源"功能创建
        </p>
      )}
    </div>
  );
}
