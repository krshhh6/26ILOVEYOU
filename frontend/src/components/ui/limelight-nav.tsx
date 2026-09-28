import React, { useState, useRef, useLayoutEffect, cloneElement } from 'react';

// --- Internal Types and Defaults ---

const DefaultHomeIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  </svg>
);
const DefaultCompassIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <path d="m16.24 7.76-2.12 6.36-6.36 2.12 2.12-6.36 6.36-2.12z" />
  </svg>
);
const DefaultBellIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
  </svg>
);

export type NavItem = {
  id: string | number;
  icon: React.ReactElement<{ className?: string }>;
  label?: string;
  badge?: string | number;
  onClick?: () => void;
};

const defaultNavItems: NavItem[] = [
  { id: 'default-home', icon: <DefaultHomeIcon />, label: 'Home' },
  { id: 'default-explore', icon: <DefaultCompassIcon />, label: 'Explore' },
  { id: 'default-notifications', icon: <DefaultBellIcon />, label: 'Notifications' },
];

export type LimelightNavProps = {
  items?: NavItem[];
  defaultActiveIndex?: number;
  activeIndex?: number;
  onTabChange?: (index: number) => void;
  className?: string;
  limelightClassName?: string;
  iconContainerClassName?: string;
  iconClassName?: string;
};

/**
 * An adaptive-width navigation bar with a "limelight" effect that highlights the active item.
 */
export const LimelightNav = ({
  items = defaultNavItems,
  defaultActiveIndex = 0,
  activeIndex: controlledActiveIndex,
  onTabChange,
  className = '',
  limelightClassName = '',
  iconContainerClassName = '',
  iconClassName = '',
}: LimelightNavProps) => {
  const [internalActiveIndex, setInternalActiveIndex] = useState(defaultActiveIndex);
  const activeIndex = controlledActiveIndex !== undefined ? controlledActiveIndex : internalActiveIndex;
  const [isReady, setIsReady] = useState(false);
  const navItemRefs = useRef<(HTMLAnchorElement | null)[]>([]);
  const limelightRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    if (items.length === 0) return;

    const updatePosition = () => {
      const limelight = limelightRef.current;
      const activeItem = navItemRefs.current[activeIndex];

      if (limelight && activeItem) {
        const newLeft = activeItem.offsetLeft + activeItem.offsetWidth / 2 - limelight.offsetWidth / 2;
        limelight.style.left = `${newLeft}px`;

        if (!isReady) {
          setIsReady(true);
        }
      }
    };

    updatePosition();
    const rafId = requestAnimationFrame(updatePosition);
    return () => cancelAnimationFrame(rafId);
  }, [activeIndex, isReady, items]);

  if (items.length === 0) {
    return null;
  }

  const handleItemClick = (index: number, itemOnClick?: () => void) => {
    setInternalActiveIndex(index);
    onTabChange?.(index);
    itemOnClick?.();
  };

  return (
    <nav className={`relative inline-flex items-center h-16 rounded-2xl bg-card text-foreground border px-2 ${className}`}>
      {items.map(({ id, icon, label, badge, onClick }, index) => {
        const isActive = activeIndex === index;
        return (
          <a
            key={id}
            ref={(el) => {
              navItemRefs.current[index] = el;
            }}
            className={`relative z-20 flex-1 flex flex-col items-center justify-center h-full cursor-pointer py-1 select-none transition-all ${iconContainerClassName}`}
            onClick={() => handleItemClick(index, onClick)}
            aria-label={label}
          >
            <div className="relative flex items-center justify-center">
              {cloneElement(icon, {
                className: `w-5 h-5 transition-transform duration-300 ease-out ${
                  isActive ? 'text-black opacity-100 scale-110' : 'text-slate-400 opacity-60'
                } ${icon.props.className || ''} ${iconClassName || ''}`,
              })}
              {badge !== undefined && (
                <span className="absolute -top-1.5 -right-3 bg-rose-500 text-white text-[9px] font-bold px-1.5 py-0.2 rounded-full leading-none shadow-sm pointer-events-none">
                  {badge}
                </span>
              )}
            </div>
            {label && (
              <span
                className={`text-[10px] mt-1 tracking-tight transition-colors duration-200 ${
                  isActive ? 'font-bold text-black' : 'font-medium text-slate-500'
                }`}
              >
                {label}
              </span>
            )}
          </a>
        );
      })}

      {/* Black Limelight Indicator Bar with Luminous Light Beam */}
      <div
        ref={limelightRef}
        className={`absolute top-0 z-10 w-12 h-[3.5px] rounded-full bg-black shadow-[0_4px_12px_rgba(0,0,0,0.35)] pointer-events-none ${limelightClassName}`}
        style={{
          left: '-999px',
          transition: isReady ? 'left 0.35s cubic-bezier(0.25, 1, 0.5, 1)' : 'none',
        }}
      >
        {/* Soft luminous spotlight light beam shining downward */}
        <div
          className="absolute left-[-40%] top-[3.5px] w-[180%] h-14 pointer-events-none"
          style={{
            clipPath: 'polygon(12% 100%, 30% 0%, 70% 0%, 88% 100%)',
            background: 'linear-gradient(180deg, rgba(0, 0, 0, 0.12) 0%, rgba(0, 0, 0, 0.04) 45%, rgba(0, 0, 0, 0) 100%)',
          }}
        />
        {/* Subtle ambient light glow highlight at base */}
        <div
          className="absolute -top-1 left-[-20%] w-[140%] h-4 pointer-events-none opacity-30 blur-[3px]"
          style={{
            background: 'radial-gradient(circle, rgba(0, 0, 0, 0.3) 0%, transparent 70%)',
          }}
        />
      </div>
    </nav>
  );
};
