'use client';

interface PersonalizationBannerProps {
  rationale?: string;
  targetGaps?: string[];
  confidence?: number;
  sessionLength?: number;
  nextKpName?: string;
}

export function PersonalizationBanner({
  rationale,
  targetGaps,
  confidence,
  sessionLength,
  nextKpName,
}: PersonalizationBannerProps) {
  return (
    <details className="rounded-lg border border-brand/20 bg-brand/5 p-4" open>
      <summary className="cursor-pointer text-sm font-medium text-brand">
        个性化适配说明
      </summary>
      <div className="mt-2 space-y-1.5 text-xs text-text-secondary">
        {rationale && <p>{rationale}</p>}
        {nextKpName && (
          <p>
            下一个推荐知识点：<span className="font-medium text-text-primary">{nextKpName}</span>
          </p>
        )}
        {targetGaps && targetGaps.length > 0 && (
          <p>
            针对薄弱环节：{targetGaps.join('、')}
          </p>
        )}
        {sessionLength && (
          <p>建议学习时长：约 {sessionLength} 分钟</p>
        )}
        {confidence !== undefined && (
          <div className="flex items-center gap-2 mt-1">
            <span>个性化置信度：</span>
            <div className="flex-1 h-1.5 rounded-full bg-brand/20 max-w-24">
              <div
                className="h-full rounded-full bg-brand"
                style={{ width: `${Math.round(confidence * 100)}%` }}
              />
            </div>
            <span>{Math.round(confidence * 100)}%</span>
          </div>
        )}
        {!rationale && !nextKpName && (
          <p>本材料根据您的学习画像进行了个性化调整</p>
        )}
      </div>
    </details>
  );
}
