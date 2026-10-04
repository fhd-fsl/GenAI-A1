import React from 'react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'universal-restoration', label: 'Universal Restoration' },
    { id: 'hard-routed-restoration', label: 'Hard-Routed Restoration' },
    { id: 'soft-mixture-of-experts', label: 'Soft Mixture-of-Experts' },
    { id: 'face-to-sketch', label: 'Face-to-Sketch' }
  ];

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-72 bg-surface-container-low/70 backdrop-blur-xl z-50 flex flex-col justify-between py-space-xl px-space-lg">
      <div className="flex flex-col gap-space-xl">
        <div className="flex items-center gap-space-md px-space-xs">
          <img alt="Brand logo." className="h-8 w-auto object-contain" src="https://lh3.googleusercontent.com/aida/AEtjO1Ub75VIIma2OLVPFfitsvXrNYuTMtKYrJ-iLzBSwEE2oFUD6x2vFi_N55GzNsOZHgSz-CZiMo0iiAwXH09yz6tDR7H645arrvb2BSAm3gymcnXirRCySkZsilIqAS7tduNIBgCRQPP1DY5BYWNbGBRjT_JpRR9FZhxyl79ojFg7Hxaie9K8i138oXZrgEn3tzokv-c9pwCTpWLhW8gqQOmmDeF9krhaJkrtch_gs1mlCmtoKTqyCLD1bLHB" />
          <span className="font-headline-md text-headline-md text-on-surface tracking-tight font-semibold">RestoreAI</span>
        </div>
        <nav className="flex flex-col gap-space-xs">
          {tabs.map((tab) => (
            <a
              key={tab.id}
              href="#"
              onClick={(e) => { e.preventDefault(); setActiveTab(tab.id); }}
              className={`px-space-md py-space-sm rounded-lg font-body-md text-body-md transition-colors duration-150 ${
                activeTab === tab.id
                  ? 'bg-surface-container text-primary font-medium'
                  : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
              }`}
            >
              {tab.label}
            </a>
          ))}
        </nav>
      </div>
      <div className="flex items-center justify-between px-space-xs pt-space-md">
        <span className="font-label-sm text-label-sm text-outline tracking-wider uppercase font-semibold">v2.4 Core</span>
        <span className="h-2 w-2 rounded-full bg-secondary-container"></span>
      </div>
    </aside>
  );
}
