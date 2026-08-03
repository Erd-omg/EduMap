'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getUserId } from '@/lib/user-id';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
const PROFILE_BASE = process.env.NEXT_PUBLIC_PROFILE_BASE || API_BASE;

export default function SettingsPage() {
  const [displayName, setDisplayName] = useState('');
  const [interactionStyle, setInteractionStyle] = useState('textual');
  const [sessionLength, setSessionLength] = useState(20);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load profile on mount
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const res = await fetch(`${PROFILE_BASE}/api/v1/profiles/${getUserId()}`);
        if (res.ok) {
          const data = await res.json();
          const p = data.profile || data;
          if (p.display_name) setDisplayName(p.display_name);
          if (p.interaction_style) {
            const topStyle = Object.entries(p.interaction_style)
              .sort(([, a], [, b]) => (b as number) - (a as number));
            if (topStyle.length > 0) setInteractionStyle(topStyle[0][0] as string);
          }
          if (p.focus_characteristics?.recommended_session_length) {
            setSessionLength(p.focus_characteristics.recommended_session_length);
          }
        }
      } catch {
        // Use defaults
      }
      setIsLoaded(true);
    };
    loadProfile();
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    setSaveMessage(null);
    try {
      await fetch(`${PROFILE_BASE}/api/v1/profiles/${getUserId()}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          display_name: displayName,
          interaction_style: {
            [interactionStyle]: 1.0,
            ...(interactionStyle !== 'visual' ? { visual: 0.3 } : {}),
            ...(interactionStyle !== 'textual' ? { textual: 0.3 } : {}),
            ...(interactionStyle !== 'interactive' ? { interactive: 0.3 } : {}),
          },
          focus_characteristics: {
            recommended_session_length: sessionLength,
          },
        }),
      });
      setSaveMessage('✅ 设置已保存');
      setTimeout(() => setSaveMessage(null), 3000);
    } catch {
      setSaveMessage('❌ 保存失败，请检查服务是否运行');
    } finally {
      setIsSaving(false);
    }
  };

  if (!isLoaded) {
    return (
      <div className="container mx-auto p-6">
        <div className="animate-pulse space-y-4 max-w-2xl mx-auto">
          <div className="h-8 rounded bg-bg-secondary w-1/3" />
          <div className="h-64 rounded-xl bg-bg-secondary" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">⚙️ 用户设置</h1>
            <p className="text-sm text-text-secondary mt-1">管理个人资料和学习偏好</p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Profile section */}
          <section className="rounded-xl border border-border bg-bg-card p-6">
            <h2 className="text-base font-semibold text-text-primary mb-4">个人资料</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1.5">
                  显示名称
                </label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="输入你的名称"
                  className="w-full rounded-lg border border-border px-4 py-2.5 text-sm text-text-primary
                             focus:border-brand focus:ring-1 focus:ring-brand outline-none"
                />
                <p className="mt-1 text-xs text-text-light">该名称将显示在个人画像和学习记录中</p>
              </div>
            </div>
          </section>

          {/* Learning preferences */}
          <section className="rounded-xl border border-border bg-bg-card p-6">
            <h2 className="text-base font-semibold text-text-primary mb-4">学习偏好</h2>
            <div className="space-y-5">
              {/* Interaction style */}
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">
                  学习风格
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { value: 'visual', label: '视觉型', icon: '👁️', desc: '偏好图表和可视化内容' },
                    { value: 'textual', label: '文字型', icon: '📖', desc: '偏好文字讲解和阅读' },
                    { value: 'interactive', label: '交互型', icon: '🖱️', desc: '偏好练习和代码实践' },
                  ].map((style) => (
                    <button
                      key={style.value}
                      onClick={() => setInteractionStyle(style.value)}
                      className={`rounded-lg border p-3 text-left transition-colors ${
                        interactionStyle === style.value
                          ? 'border-brand bg-brand/5'
                          : 'border-border hover:border-brand/30 hover:bg-bg-secondary'
                      }`}
                    >
                      <span className="text-lg block mb-1">{style.icon}</span>
                      <p className="text-sm font-medium text-text-primary">{style.label}</p>
                      <p className="text-[10px] text-text-light mt-0.5">{style.desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Session length */}
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1.5">
                  建议学习时长：{sessionLength} 分钟
                </label>
                <input
                  type="range"
                  min={5}
                  max={60}
                  step={5}
                  value={sessionLength}
                  onChange={(e) => setSessionLength(Number(e.target.value))}
                  className="w-full accent-brand"
                />
                <div className="flex justify-between text-[10px] text-text-light mt-0.5">
                  <span>5 分钟</span>
                  <span>60 分钟</span>
                </div>
              </div>
            </div>
          </section>

          {/* Notifications */}
          <section className="rounded-xl border border-border bg-bg-card p-6">
            <h2 className="text-base font-semibold text-text-primary mb-4">通知偏好</h2>
            <div className="space-y-3">
              <label className="flex items-center justify-between p-3 rounded-lg hover:bg-bg-secondary transition-colors cursor-pointer">
                <div>
                  <p className="text-sm font-medium text-text-primary">学习提醒</p>
                  <p className="text-xs text-text-light">接收遗忘曲线复习提醒和推荐计划</p>
                </div>
                <div
                  onClick={(e) => {
                    e.preventDefault();
                    setNotificationsEnabled(!notificationsEnabled);
                  }}
                  className={`relative w-10 h-5 rounded-full transition-colors cursor-pointer ${
                    notificationsEnabled ? 'bg-brand' : 'bg-border'
                  }`}
                >
                  <div
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                      notificationsEnabled ? 'translate-x-5' : 'translate-x-0.5'
                    }`}
                  />
                </div>
              </label>
            </div>
          </section>

          {/* Data management */}
          <section className="rounded-xl border border-border bg-bg-card p-6">
            <h2 className="text-base font-semibold text-text-primary mb-4">数据管理</h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-lg hover:bg-bg-secondary transition-colors">
                <div>
                  <p className="text-sm font-medium text-text-primary">导出学习数据</p>
                  <p className="text-xs text-text-light">下载你的学习记录和画像数据</p>
                </div>
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch(`${PROFILE_BASE}/api/v1/profiles/${getUserId()}`);
                      if (res.ok) {
                        const data = await res.json();
                        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `edumap-profile-${Date.now()}.json`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }
                    } catch {
                      setSaveMessage('❌ 导出失败');
                    }
                  }}
                  className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
                >
                  导出
                </button>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg hover:bg-bg-secondary transition-colors">
                <div>
                  <p className="text-sm font-medium text-text-danger">清除所有数据</p>
                  <p className="text-xs text-text-light">删除本地存储的所有学习数据</p>
                </div>
                <button
                  onClick={() => {
                    if (confirm('确定要清除所有本地数据吗？此操作不可撤销。')) {
                      localStorage.clear();
                      setSaveMessage('✅ 本地数据已清除，即将刷新页面');
                      setTimeout(() => window.location.reload(), 1500);
                    }
                  }}
                  className="rounded-lg border border-danger/30 px-3 py-1.5 text-xs font-medium text-danger hover:bg-danger-bg transition-colors"
                >
                  清除
                </button>
              </div>
            </div>
          </section>

          {/* Save button */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="rounded-lg bg-brand px-6 py-2.5 text-sm font-medium text-white hover:bg-brand-hover disabled:opacity-50 transition-colors"
            >
              {isSaving ? '保存中...' : '保存设置'}
            </button>
            {saveMessage && (
              <span className="text-sm text-text-secondary">{saveMessage}</span>
            )}
            <Link
              href="/dashboard"
              className="ml-auto text-sm text-text-light hover:text-text-primary transition-colors"
            >
              返回仪表盘
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
