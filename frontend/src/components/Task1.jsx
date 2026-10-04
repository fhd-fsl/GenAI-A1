import React, { useState } from 'react';
import { restoreImage } from '../api';

export default function Task1() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [corruptionType, setCorruptionType] = useState('gaussian_blur');
  const [severity, setSeverity] = useState('medium');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const handleRun = async () => {
    if (!file) return;
    setLoading(true);
    try {
      let params = {};
      if (corruptionType === 'gaussian_blur') {
        params = severity === 'low' ? { kernel_size: 3, sigma: 1.0 } : severity === 'medium' ? { kernel_size: 5, sigma: 1.5 } : { kernel_size: 7, sigma: 2.5 };
      } else if (corruptionType === 'salt_and_pepper') {
        params = severity === 'low' ? { probability: 0.05 } : severity === 'medium' ? { probability: 0.1 } : { probability: 0.2 };
      } else if (corruptionType === 'rectangular_occlusion') {
        params = severity === 'low' ? { num_rects: 1, area_pct: 0.1 } : severity === 'medium' ? { num_rects: 2, area_pct: 0.2 } : { num_rects: 3, area_pct: 0.3 };
      }

      const data = await restoreImage(file, corruptionType, params);
      setResult(data);
    } catch (err) {
      console.error(err);
      alert('Error restoring image');
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col gap-space-lg w-full max-w-7xl mx-auto">
      {/* Workspace Title */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-space-md mb-space-lg">
        <div className="flex flex-col gap-space-xs">
          <div className="flex items-center gap-space-sm">
            <span className="font-label-sm text-label-sm text-primary tracking-widest uppercase font-semibold">Pipeline 01</span>
            <span className="w-1.5 h-1.5 rounded-full bg-outline-variant"></span>
            <span className="font-label-sm text-label-sm text-on-surface-variant font-medium">Single Model Autoencoder</span>
          </div>
          <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight font-semibold">Universal Restoration</h1>
        </div>
      </div>

      {/* Top Control Bar */}
      <div className="w-full bg-surface-container rounded-xl shadow-md p-space-md mb-space-lg">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-md items-center">
          <div className="lg:col-span-4 relative group">
            <label className="flex items-center justify-between px-space-md py-space-sm rounded-lg bg-surface-container-low hover:bg-surface-container-high transition-colors duration-150 cursor-pointer border border-transparent hover:border-surface-container-high">
              <div className="flex items-center gap-space-sm min-w-0">
                <div className="w-8 h-8 rounded-lg bg-surface-container-highest flex items-center justify-center shrink-0 text-primary">
                  <span className="material-symbols-outlined text-[20px]">add_photo_alternate</span>
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="font-body-md text-body-md text-on-surface font-medium truncate">{file ? file.name : "Drop source or browse"}</span>
                  <span className="font-label-sm text-label-sm text-outline truncate">RGB Image File</span>
                </div>
              </div>
              <span className="material-symbols-outlined text-outline group-hover:text-primary transition-colors text-[20px] ml-space-sm shrink-0">upload_file</span>
              <input accept="image/*" className="sr-only" type="file" onChange={handleFileChange} />
            </label>
          </div>

          <div className="lg:col-span-5 flex items-center gap-space-sm bg-surface-container-low px-space-md py-space-sm rounded-lg">
            <span className="material-symbols-outlined text-outline text-[20px] shrink-0">tune</span>
            <div className="flex flex-col w-1/2 border-r border-surface-container-high pr-space-sm mr-space-sm">
              <span className="font-label-sm text-label-sm text-outline uppercase font-semibold tracking-wider">Corruption</span>
              <select className="bg-transparent font-body-md text-body-md text-on-surface focus:outline-none cursor-pointer w-full" value={corruptionType} onChange={e => setCorruptionType(e.target.value)}>
                <option className="bg-surface-container-high" value="clean">Clean (None)</option>
                <option className="bg-surface-container-high" value="gaussian_blur">Gaussian Blur</option>
                <option className="bg-surface-container-high" value="salt_and_pepper">Salt & Pepper</option>
                <option className="bg-surface-container-high" value="rectangular_occlusion">Occlusion</option>
              </select>
            </div>
            <div className="flex flex-col w-1/2">
              <span className="font-label-sm text-label-sm text-outline uppercase font-semibold tracking-wider">Severity</span>
              <select className="bg-transparent font-body-md text-body-md text-on-surface focus:outline-none cursor-pointer w-full" value={severity} onChange={e => setSeverity(e.target.value)} disabled={corruptionType === 'clean'}>
                <option className="bg-surface-container-high" value="low">Low</option>
                <option className="bg-surface-container-high" value="medium">Medium</option>
                <option className="bg-surface-container-high" value="high">High</option>
              </select>
            </div>
          </div>

          <div className="lg:col-span-3 flex items-center justify-end">
            <button onClick={handleRun} disabled={loading || !file} className="w-full py-space-md px-space-lg rounded-lg bg-primary-container text-on-primary-container font-headline-md text-headline-md tracking-tight hover:brightness-110 active:scale-[0.99] transition-all flex items-center justify-center gap-space-sm shadow-md">
              <span className={`material-symbols-outlined text-[20px] ${loading ? 'animate-spin' : ''}`}>auto_fix_high</span>
              <span className="font-semibold text-body-lg">Restore Image</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Viewport */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-space-lg mb-space-lg">
        <div className="flex flex-col bg-surface-container rounded-xl shadow-lg overflow-hidden border border-surface-container-high">
          <div className="px-space-md py-space-sm flex items-center justify-between bg-surface-container-high">
            <div className="flex items-center gap-space-sm">
              <span className="w-2 h-2 rounded-full bg-error"></span>
              <span className="font-label-md text-label-md uppercase tracking-wider font-semibold text-on-surface">Input Image</span>
            </div>
          </div>
          <div className="relative w-full aspect-[4/3] bg-surface-container-lowest flex items-center justify-center p-space-sm overflow-hidden">
            {result ? (
              <img className="w-full h-full object-cover rounded-lg" src={result.input_image} alt="Input" />
            ) : preview ? (
              <img className="w-full h-full object-cover rounded-lg opacity-50" src={preview} alt="Preview" />
            ) : <span className="text-on-surface-variant">Awaiting Upload</span>}
          </div>
        </div>
        <div className="flex flex-col bg-surface-container rounded-xl shadow-lg overflow-hidden border border-surface-container-high">
          <div className="px-space-md py-space-sm flex items-center justify-between bg-surface-container-high">
            <div className="flex items-center gap-space-sm">
              <span className="w-2 h-2 rounded-full bg-secondary"></span>
              <span className="font-label-md text-label-md uppercase tracking-wider font-semibold text-on-surface">Restored Image</span>
            </div>
          </div>
          <div className="relative w-full aspect-[4/3] bg-surface-container-lowest flex items-center justify-center p-space-sm overflow-hidden">
            {result && <img className="w-full h-full object-cover rounded-lg" src={result.output_image} alt="Output" />}
          </div>
        </div>
      </div>

      {/* Timing and Metrics Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-lg items-stretch">
        <div className="lg:col-span-8 bg-surface-container rounded-xl shadow-md p-space-lg flex flex-col justify-center border border-surface-container-high">
            <div className="flex items-center gap-space-md">
                <div className="p-space-sm bg-surface-container-high rounded-full">
                    <span className="material-symbols-outlined text-primary text-[24px]">model_training</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-on-surface font-semibold text-body-lg">Universal Diffusion Model v1</span>
                    <span className="text-on-surface-variant text-label-md">Trained on all corruption types concurrently</span>
                </div>
            </div>
        </div>
        <div className="lg:col-span-4 flex flex-col gap-space-md justify-between">
          <div className="bg-surface-container rounded-xl shadow-md p-space-lg flex-1 flex flex-col justify-between border border-surface-container-high">
            <div className="flex items-center justify-between mb-space-sm">
              <span className="font-label-md text-label-md uppercase tracking-wider font-semibold text-on-surface">Execution Timing</span>
              {result && (
                <span className="flex items-center gap-space-xs text-label-sm font-label-sm text-secondary">
                  <span className="w-2 h-2 rounded-full bg-secondary animate-ping"></span>
                  Synchronized
                </span>
              )}
            </div>
            <div className="my-space-md">
              <div className="flex items-baseline gap-space-sm">
                <span className="font-display text-display text-primary font-bold tracking-tight">
                  {result ? result.inference_time_ms.toFixed(1) : "0.0"}
                </span>
                <span className="font-headline-md text-headline-md text-primary font-light">ms</span>
              </div>
              <span className="font-label-sm text-label-sm text-on-surface-variant">Total Pipeline Round-Trip</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
