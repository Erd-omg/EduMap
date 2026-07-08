import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-8 p-8">
      <h1 className="text-4xl font-bold tracking-tight">智图 EduMap</h1>
      <p className="max-w-md text-center text-lg text-gray-600">
        多智能体驱动的个性化学习平台
      </p>
      <div className="flex gap-4">
        <Link
          href="/chat"
          className="rounded-lg bg-primary px-6 py-3 text-white hover:bg-primary-dark"
        >
          开始学习
        </Link>
        <Link
          href="/profile"
          className="rounded-lg border px-6 py-3 text-gray-700 hover:bg-gray-100"
        >
          查看画像
        </Link>
      </div>
    </div>
  );
}
