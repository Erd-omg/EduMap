'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

interface CourseInfo {
  id: string;
  name: string;
  description: string;
  difficulty: number;
}

export function CourseSelector() {
  const [courses, setCourses] = useState<CourseInfo[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const pathname = usePathname();

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/kg/courses`);
        if (res.ok) {
          const data = await res.json();
          setCourses(data.courses || []);
        }
      } catch {
        // Ignore — fall back to default
      } finally {
        setIsLoading(false);
      }
    };
    fetchCourses();
  }, []);

  // Determine current course from URL
  const currentCourseId = pathname.startsWith('/learn/') ? pathname.split('/')[2] : null;
  const currentCourse = courses.find((c) => c.id === currentCourseId);

  if (courses.length === 0 && !isLoading) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-text-primary hover:bg-bg-secondary transition-colors"
      >
        <span>📚</span>
        <span className="max-w-[120px] truncate">
          {currentCourse?.name || (isLoading ? '加载中...' : '选择课程')}
        </span>
        <svg className="w-3 h-3 text-text-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full left-0 mt-1 z-50 w-64 rounded-xl border border-border bg-white shadow-lg">
            <div className="p-2 space-y-0.5">
              {courses.length === 0 ? (
                <p className="px-3 py-4 text-xs text-text-light text-center">暂无课程</p>
              ) : (
                courses.map((course) => {
                  const isActive = pathname === `/learn/${course.id}`;
                  return (
                    <Link
                      key={course.id}
                      href={`/learn/${course.id}`}
                      onClick={() => setIsOpen(false)}
                      className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors ${
                        isActive
                          ? 'bg-brand/5 text-brand font-medium'
                          : 'text-text-secondary hover:bg-bg-secondary'
                      }`}
                    >
                      <span className="text-base">
                        {course.difficulty <= 2 ? '🌱' : course.difficulty <= 3 ? '🌿' : '🌳'}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className={`truncate ${isActive ? 'text-brand' : 'text-text-primary'}`}>
                          {course.name}
                        </p>
                        <p className="text-[10px] text-text-light truncate">{course.description}</p>
                      </div>
                      <span className="text-[10px] text-text-light">Lv.{course.difficulty}</span>
                    </Link>
                  );
                })
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
