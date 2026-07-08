'use client';
export function ProfileCard() {
  return (
    <div className="rounded-xl border bg-white p-6">
      <h2 className="mb-4 text-lg font-semibold">6 维画像概览</h2>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
        {['知识基础', '学习能力', '学习动机', '知识覆盖率', '交互偏好', '专注力'].map(
          (dim) => (
            <div key={dim} className="rounded-lg bg-gray-50 p-3 text-center">
              <p className="text-sm text-gray-500">{dim}</p>
              <p className="text-lg font-bold text-primary">--</p>
            </div>
          ),
        )}
      </div>
    </div>
  );
}
