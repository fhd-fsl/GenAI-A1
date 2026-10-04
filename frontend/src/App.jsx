import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Task1 from './components/Task1';
import Task2 from './components/Task2';
import Task3 from './components/Task3';
import Task4 from './components/Task4';

function App() {
  const [activeTab, setActiveTab] = useState('universal-restoration');
  
  return (
    <>
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="pl-72 min-h-screen flex flex-col">
        <header className="fixed top-0 left-72 right-0 z-40 bg-surface-container-lowest/80 backdrop-blur-xl shadow-[0_1px_8px_rgba(0,0,0,0.04)]">
          <div className="h-16 w-full px-space-xl flex items-center justify-between">
            <div className="flex items-center gap-space-md">
              <img alt="Brand logo." className="h-8 w-auto object-contain" src="https://lh3.googleusercontent.com/aida/AEtjO1Ub75VIIma2OLVPFfitsvXrNYuTMtKYrJ-iLzBSwEE2oFUD6x2vFi_N55GzNsOZHgSz-CZiMo0iiAwXH09yz6tDR7H645arrvb2BSAm3gymcnXirRCySkZsilIqAS7tduNIBgCRQPP1DY5BYWNbGBRjT_JpRR9FZhxyl79ojFg7Hxaie9K8i138oXZrgEn3tzokv-c9pwCTpWLhW8gqQOmmDeF9krhaJkrtch_gs1mlCmtoKTqyCLD1bLHB" />
              <span className="font-label-md text-label-md text-on-surface-variant tracking-wide uppercase">Workspace</span>
            </div>
            <div className="flex items-center gap-space-md">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                <span className="material-symbols-outlined text-on-primary text-[18px]">person</span>
              </div>
            </div>
          </div>
        </header>
        <main className="w-full pt-16 px-space-xl pb-space-xl bg-surface-container-lowest flex-1 relative">
          <div className="absolute -top-10 left-1/3 w-96 h-96 bg-primary/5 rounded-full blur-3xl pointer-events-none"></div>
          <div className="absolute top-1/2 right-10 w-80 h-80 bg-secondary/5 rounded-full blur-3xl pointer-events-none"></div>
          
          <div className="relative z-10 w-full pt-space-lg">
            {activeTab === 'universal-restoration' && <Task1 />}
            {activeTab === 'hard-routed-restoration' && <Task2 />}
            {activeTab === 'soft-mixture-of-experts' && <Task3 />}
            {activeTab === 'face-to-sketch' && <Task4 />}
          </div>
        </main>
      </div>
    </>
  );
}

export default App;
