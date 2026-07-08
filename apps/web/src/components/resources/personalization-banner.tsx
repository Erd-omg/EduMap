export function PersonalizationBanner() {
  return (
    <details className="rounded-lg border border-blue-200 bg-blue-50 p-4" open>
      <summary className="cursor-pointer text-sm font-medium text-blue-700">
        个性化适配说明
      </summary>
      <p className="mt-2 text-sm text-blue-600">
        本材料根据您的学习画像进行了个性化调整（待实现完整逻辑）
      </p>
    </details>
  );
}
