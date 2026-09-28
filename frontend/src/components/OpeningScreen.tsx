import React, { useState, useEffect, useRef } from 'react';
import { Globe } from '@/components/ui/globe';

interface OpeningScreenProps {
  onComplete: () => void;
  durationMs?: number; // default 3200ms (3.2 seconds)
}

export const OpeningScreen: React.FC<OpeningScreenProps> = ({
  onComplete,
  durationMs = 3200,
}) => {
  const [phase, setPhase] = useState<'displaying' | 'zooming' | 'completed'>('displaying');
  const timerRef = useRef<number | null>(null);

  const triggerZoomTransition = () => {
    if (phase !== 'displaying') return;
    setPhase('zooming');
    // Smooth center zoom-in and blended crossfade to website dashboard
    setTimeout(() => {
      setPhase('completed');
      onComplete();
    }, 1200);
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      triggerZoomTransition();
    }, durationMs);

    timerRef.current = timer as unknown as number;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') {
        clearTimeout(timer);
        triggerZoomTransition();
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [durationMs]);

  if (phase === 'completed') {
    return null;
  }

  const isZooming = phase === 'zooming';

  return (
    <div
      onClick={triggerZoomTransition}
      className="fixed inset-0 z-[99999] flex items-center justify-center bg-white dark:bg-black select-none overflow-hidden cursor-pointer"
      style={{
        opacity: isZooming ? 0 : 1,
        pointerEvents: isZooming ? 'none' : 'auto',
        transition: 'opacity 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
      title="Click or press any key to enter dashboard"
    >
      {/* 16:9 Stage Container */}
      <div className="relative w-full h-full max-w-[1920px] max-h-[1080px] aspect-video flex flex-col items-center justify-center overflow-hidden">
        
        {/* Layer 1 (z-10): Spill Sense Text - 80% Visible, positioned behind the globe */}
        <div
          className="absolute left-1/2 z-10 flex items-center justify-center select-none"
          style={{
            top: '16vh',
            transform: isZooming
              ? 'translateX(-50%) scale(1.15)'
              : 'translateX(-50%) scale(1)',
            filter: isZooming ? 'blur(12px)' : 'none',
            opacity: isZooming ? 0 : 1,
            transition:
              'transform 0.9s cubic-bezier(0.16, 1, 0.3, 1), filter 0.9s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.9s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          <span
            className="pointer-events-none whitespace-nowrap bg-gradient-to-b from-black to-gray-300/80 bg-clip-text text-center font-semibold leading-none text-transparent dark:from-white dark:to-slate-900/10 tracking-tight"
            style={{
              fontSize: 'clamp(3.5rem, 8.2vw, 8.5rem)',
            }}
          >
            Spill Sense
          </span>
        </div>

        {/* Layer 2 (z-20): 70% Globe Shown From Bottom - In front of text */}
        <div
          className="absolute left-1/2 z-20 pointer-events-auto"
          style={{
            top: '33vh',
            width: 'min(94vh, 920px)',
            height: 'min(94vh, 920px)',
            transform: isZooming
              ? 'translateX(-50%) scale(3.5)'
              : 'translateX(-50%) scale(1)',
            filter: isZooming ? 'blur(24px)' : 'none',
            opacity: isZooming ? 0 : 1,
            transition:
              'transform 1.2s cubic-bezier(0.16, 1, 0.3, 1), filter 1.0s cubic-bezier(0.16, 1, 0.3, 1), opacity 1.1s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          <Globe className="!static !w-full !h-full !max-w-none" />
        </div>

        {/* Layer 3 (z-30): Ambient Radial Glow Overlay */}
        <div className="pointer-events-none absolute inset-0 z-30 h-full bg-[radial-gradient(circle_at_50%_200%,rgba(0,0,0,0.2),rgba(255,255,255,0))]" />
      </div>
    </div>
  );
};
