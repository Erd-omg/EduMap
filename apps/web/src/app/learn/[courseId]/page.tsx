export default async function CoursePage(props: { params: Promise<{ courseId: string }> }) {
  const { courseId } = await props.params;
  return (
    <div className="container mx-auto p-6">
      <h1 className="mb-6 text-2xl font-bold">课程：{courseId}</h1>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="rounded-xl border bg-white p-4" style={{ height: '500px' }}>
            <p className="text-center text-gray-500">技能树可视化区域</p>
          </div>
        </div>
        <div>
          <div className="rounded-xl border bg-white p-4">
            <p className="mb-2 text-sm font-medium">学习进度</p>
            <input type="range" min="0" max="100" className="w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
