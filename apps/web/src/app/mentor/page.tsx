'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function MentorPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/chat');
  }, [router]);

  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <p className="text-sm text-text-light">正在跳转到统一对话页面...</p>
    </div>
  );
}
