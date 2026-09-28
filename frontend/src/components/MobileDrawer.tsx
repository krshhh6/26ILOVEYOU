import React, { useEffect, useRef } from 'react';
import type { TabType } from '../types/dashboard';
import type { LiveIncident } from '../hooks/useIncidents';
import { Sidebar } from './Sidebar';

interface MobileDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: TabType;
  onSelectTab: (tab: TabType) => void;
  currentScenarioKey?: string;
  onSelectScenario?: (key: string) => void;
  onOpenSettings?: () => void;
  incidents?: LiveIncident[];
}

export const MobileDrawer: React.FC<MobileDrawerProps> = ({
  isOpen,
  onClose,
  activeTab,
  onSelectTab,
  currentScenarioKey,
  onSelectScenario,
  onOpenSettings,
  incidents,
}) => {
  const drawerRef = useRef<HTMLDivElement>(null);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Prevent body scroll when drawer is open on mobile
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleTabClick = (tab: TabType) => {
    onSelectTab(tab);
    onClose();
  };

  const handleScenarioClick = (key: string) => {
    if (onSelectScenario) {
      onSelectScenario(key);
    }
    onClose();
  };

  return (
    <div className="mobile-drawer-portal" role="dialog" aria-modal="true" aria-label="Mobile Navigation Menu">
      {/* Backdrop */}
      <div
        className="mobile-drawer-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-Up Sheet Container */}
      <div className="mobile-drawer-sheet" ref={drawerRef}>
        {/* Grab Handle */}
        <div className="mobile-drawer-handle-bar" onClick={onClose}>
          <div className="mobile-drawer-handle" />
        </div>

        {/* Header with Title and Close Button */}
        <div className="mobile-drawer-header">
          <div className="mobile-drawer-title">
            <span className="material-symbols-rounded text-cyan-400">shield</span>
            <span>SpillSense C2 Hub</span>
          </div>
          <button
            type="button"
            className="mobile-drawer-close-btn"
            onClick={onClose}
            aria-label="Close menu"
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        {/* Inner Content - Reuses Sidebar Component logic inside mobile wrapper */}
        <div className="mobile-drawer-content">
          <Sidebar
            activeTab={activeTab}
            onSelectTab={handleTabClick}
            currentScenarioKey={currentScenarioKey}
            onSelectScenario={handleScenarioClick}
            onOpenSettings={onOpenSettings}
            incidents={incidents}
          />
        </div>
      </div>
    </div>
  );
};
