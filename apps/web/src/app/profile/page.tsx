'use client';
import { ProfileCard } from '@/components/profiling/profile-card';

export default function ProfilePage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="mb-6 text-2xl font-bold">学习画像</h1>
      <ProfileCard />
    </div>
  );
}
