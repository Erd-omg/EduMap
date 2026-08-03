'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface DemoConsoleProps {
  initialMastered: number;
  initialLearning: number;
  initialNotStarted: number;
  totalCount: number;
  onProgressUpdate?: (mastered: number, learning: number, notStarted: number) => void;
}

type PlayState = 'idle' | 'playing' | 'paused';
type Speed = 1 | 2 | 4;

/**
 * DemoConsole — interactive progress simulation for demo/presentation use.
 *
 * Renders below the ProgressSlider and lets the user simulate learning
 * progress over N days at configurable speed.
 */
export function DemoConsole({
  initialMastered,
  initialLearning,
  initialNotStarted,
  totalCount,
  onProgressUpdate,
}: DemoConsoleProps) {
  const [playState, setPlayState] = useState<PlayState>('idle');
  const [currentDay, setCurrentDay] = useState(0);
  const [speed, setSpeed] = useState<Speed>(2);
  const [mastered, setMastered] = useState(initialMastered);
  const [learning, setLearning] = useState(initialLearning);
  const [notStarted, setNotStarted] = useState(initialNotStarted);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dayRef = useRef(0);
  const masteredRef = useRef(initialMastered);
  const learningRef = useRef(initialLearning);
  const notStartedRef = useRef(initialNotStarted);

  const MAX_DAYS = 7;

  // Sync external initial values when parent re-renders (in idle state)
  useEffect(() => {
    if (playState === 'idle') {
      setMastered(initialMastered);
      setLearning(initialLearning);
      setNotStarted(initialNotStarted);
    }
  }, [initialMastered, initialLearning, initialNotStarted, playState]);

  // Notify parent of current progress
  const notify = useCallback(
    (m: number, l: number, ns: number) => {
      onProgressUpdate?.(m, l, ns);
    },
    [onProgressUpdate],
  );

  // Generate next day's progress
  const advanceDay = useCallback(() => {
    const d = dayRef.current + 1;
    if (d > MAX_DAYS) {
      stopPlayback();
      return;
    }

    dayRef.current = d;
    setCurrentDay(d);

    let m = masteredRef.current;
    let l = learningRef.current;
    let ns = notStartedRef.current;

    // Simulate learning: move some not_started → learning, some learning → mastered
    // Rate increases over days to simulate acceleration
    const dayFactor = 0.15 + (d / MAX_DAYS) * 0.25; // 0.15 → 0.4

    // Learning → Mastered (with some leftover in learning)
    const toMastered = Math.max(1, Math.round(l * dayFactor * 0.6));
    const masteredToday = Math.min(toMastered, l);
    m += masteredToday;
    l -= masteredToday;

    // Not started → Learning
    const toLearning = Math.max(1, Math.round(ns * dayFactor * 0.4));
    const learningToday = Math.min(toLearning, ns);
    l += learningToday;
    ns -= learningToday;

    masteredRef.current = m;
    learningRef.current = l;
    notStartedRef.current = ns;

    setMastered(m);
    setLearning(l);
    setNotStarted(ns);
    notify(m, l, ns);
  }, [notify]);

  const stopPlayback = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPlayState('idle');
  }, []);

  const startPlayback = useCallback(() => {
    setPlayState('playing');
    // Reset to initial state before starting fresh
    const resetM = initialMastered;
    const resetL = initialLearning;
    const resetNs = initialNotStarted;
    masteredRef.current = resetM;
    learningRef.current = resetL;
    notStartedRef.current = resetNs;
    dayRef.current = 0;
    setCurrentDay(0);
    setMastered(resetM);
    setLearning(resetL);
    setNotStarted(resetNs);
    notify(resetM, resetL, resetNs);

    const intervalMs = Math.round(1000 / speed); // 1x=1000ms, 2x=500ms, 4x=250ms
    intervalRef.current = setInterval(advanceDay, intervalMs);
  }, [initialMastered, initialLearning, initialNotStarted, speed, advanceDay, notify]);

  const pausePlayback = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPlayState('paused');
  }, []);

  const resumePlayback = useCallback(() => {
    setPlayState('playing');
    const intervalMs = Math.round(1000 / speed);
    intervalRef.current = setInterval(advanceDay, intervalMs);
  }, [speed, advanceDay]);

  const resetPlayback = useCallback(() => {
    stopPlayback();
    masteredRef.current = initialMastered;
    learningRef.current = initialLearning;
    notStartedRef.current = initialNotStarted;
    dayRef.current = 0;
    setCurrentDay(0);
    setMastered(initialMastered);
    setLearning(initialLearning);
    setNotStarted(initialNotStarted);
    notify(initialMastered, initialLearning, initialNotStarted);
  }, [initialMastered, initialLearning, initialNotStarted, stopPlayback, notify]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Re-create interval when speed changes mid-playback
  useEffect(() => {
    if (playState === 'playing') {
      if (intervalRef.current) clearInterval(intervalRef.current);
      const intervalMs = Math.round(1000 / speed);
      intervalRef.current = setInterval(advanceDay, intervalMs);
    }
  }, [speed, playState, advanceDay]);

  const isPlaying = playState === 'playing';
  const progressPct = totalCount > 0 ? Math.round((mastered / totalCount) * 100) : 0;

  return (
    <div className="mt-4 rounded-lg border border-border bg-bg-card p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-text-primary">
          🎮 演示模拟
        </h4>
        <span className="text-xs text-text-light">
          Day {currentDay}/{MAX_DAYS} · {progressPct}% 完成
        </span>
      </div>

      {/* Timeline slider */}
      <div className="relative">
        <input
          type="range"
          min={0}
          max={MAX_DAYS}
          value={currentDay}
          readOnly
          className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-border accent-brand"
          aria-label="模拟进度时间线"
        />
        <div className="flex justify-between mt-1 text-[10px] text-text-light">
          <span>Day 1</span>
          <span>Day 4</span>
          <span>Day 7</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        {/* Play/Pause */}
        {!isPlaying ? (
          <button
            onClick={playState === 'paused' ? resumePlayback : startPlayback}
            className="flex items-center gap-1.5 rounded-lg bg-brand px-4 py-1.5 text-sm text-white hover:bg-brand-hover transition-colors"
            disabled={currentDay >= MAX_DAYS && playState !== 'paused'}
          >
            <span>{playState === 'paused' ? '▶' : '▶'}</span>
            <span>{playState === 'paused' ? '继续' : '开始'}</span>
          </button>
        ) : (
          <button
            onClick={pausePlayback}
            className="flex items-center gap-1.5 rounded-lg bg-warning px-4 py-1.5 text-sm text-white"
          >
            <span>⏸</span>
            <span>暂停</span>
          </button>
        )}

        {/* Reset */}
        <button
          onClick={resetPlayback}
          className="rounded-lg border border-border px-3 py-1.5 text-sm text-text-secondary hover:bg-bg-secondary transition-colors"
          disabled={playState === 'idle'}
        >
          ↺ 重置
        </button>

        {/* Speed */}
        <div className="flex items-center gap-1">
          {([1, 2, 4] as Speed[]).map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`rounded px-2 py-1 text-xs transition-colors ${
                speed === s
                  ? 'bg-brand text-white'
                  : 'border border-border text-text-secondary hover:bg-bg-secondary'
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      <div className="flex items-center justify-center gap-6 text-xs">
        <span className="text-success">已掌握 {mastered}</span>
        <span className="text-warning">学习中 {learning}</span>
        <span className="text-text-light">未开始 {notStarted}</span>
      </div>
    </div>
  );
}
