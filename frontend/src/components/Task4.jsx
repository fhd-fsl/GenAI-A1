import React, { useState, useRef, useEffect } from 'react';
import { generateSketch } from '../api';

export default function Task4() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [styleIdx, setStyleIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [stream, setStream] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const startWebcam = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(s);
      setWebcamActive(true);
    } catch (err) {
      console.error(err);
      alert('Could not access webcam. Please allow permissions.');
    }
  };

  useEffect(() => {
    if (webcamActive && videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [webcamActive, stream]);

  const captureWebcam = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      // Note: we draw it as-is, but if mirrored we'd flip the context, standard is fine for ML.
      context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      
      canvasRef.current.toBlob((blob) => {
        const capturedFile = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
        setFile(capturedFile);
        setPreview(URL.createObjectURL(capturedFile));
        setResult(null);
        stopWebcam();
      }, 'image/jpeg');
    }
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setWebcamActive(false);
  };

  const handleRun = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const data = await generateSketch(file, styleIdx);
      setResult(data);
    } catch (err) {
      console.error(err);
      alert('Error generating sketch');
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col gap-space-lg w-full max-w-7xl mx-auto">
      {/* Workspace Title */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-space-md mb-space-lg">
        <div className="flex flex-col gap-space-xs">
          <div className="flex items-center gap-space-sm">
            <span className="font-label-sm text-label-sm text-primary tracking-widest uppercase font-semibold">Pipeline 04</span>
            <span className="w-1.5 h-1.5 rounded-full bg-outline-variant"></span>
            <span className="font-label-sm text-label-sm text-on-surface-variant font-medium">Paired Translation</span>
          </div>
          <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight font-semibold">Face-to-Sketch Synthesis</h1>
        </div>
      </div>

      {/* Top Control Bar */}
      <div className="w-full bg-surface-container rounded-xl shadow-md p-space-md mb-space-lg border border-surface-container-high">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-md items-center">
          <div className="lg:col-span-4 flex gap-space-sm h-full items-stretch">
            <label className="flex-1 flex items-center justify-between px-space-md py-space-sm rounded-lg bg-surface-container-low hover:bg-surface-container-high transition-colors duration-150 cursor-pointer border border-transparent hover:border-surface-container-high">
              <div className="flex items-center gap-space-sm min-w-0">
                <div className="w-8 h-8 rounded-lg bg-surface-container-highest flex items-center justify-center shrink-0 text-primary">
                  <span className="material-symbols-outlined text-[20px]">add_photo_alternate</span>
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="font-body-md text-body-md text-on-surface font-medium truncate">{file ? file.name : "Drop source or browse"}</span>
                  <span className="font-label-sm text-label-sm text-outline truncate">RGB Face Photograph</span>
                </div>
              </div>
              <input accept="image/*" className="sr-only" type="file" onChange={handleFileChange} />
            </label>
            <button onClick={startWebcam} title="Use Webcam" className="px-space-md flex flex-col items-center justify-center rounded-lg bg-surface-container-low hover:bg-surface-container-high text-primary transition-colors border border-transparent hover:border-surface-container-high shrink-0">
              <span className="material-symbols-outlined text-[20px]">photo_camera</span>
            </button>
          </div>

          <div className="lg:col-span-5 flex items-center gap-space-sm bg-surface-container-low px-space-md py-space-sm rounded-lg">
            <span className="material-symbols-outlined text-outline text-[20px] shrink-0">palette</span>
            <div className="flex flex-col w-full pr-space-sm mr-space-sm">
              <span className="font-label-sm text-label-sm text-outline uppercase font-semibold tracking-wider">Sketch Style Target</span>
              <select className="bg-transparent font-body-md text-body-md text-on-surface focus:outline-none cursor-pointer w-full" value={styleIdx} onChange={e => setStyleIdx(Number(e.target.value))}>
                <option className="bg-surface-container-high" value={0}>Style 1 (Standard Pencil)</option>
                <option className="bg-surface-container-high" value={1}>Style 2 (Charcoal Shading)</option>
                <option className="bg-surface-container-high" value={2}>Style 3 (Ink Wash)</option>
              </select>
            </div>
          </div>

          <div className="lg:col-span-3 flex items-center justify-end h-full">
            <button onClick={handleRun} disabled={loading || !file || webcamActive} className="w-full h-full py-space-md px-space-lg rounded-lg bg-primary-container text-on-primary-container font-headline-md text-headline-md tracking-tight hover:brightness-110 active:scale-[0.99] transition-all flex items-center justify-center gap-space-sm shadow-md disabled:opacity-50 disabled:pointer-events-none">
              <span className={`material-symbols-outlined text-[20px] ${loading ? 'animate-spin' : ''}`}>edit_square</span>
              <span className="font-semibold text-body-lg">Synthesize</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Viewport */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-space-lg mb-space-lg">
        <div className="flex flex-col bg-surface-container rounded-xl shadow-lg overflow-hidden border border-surface-container-high">
          <div className="px-space-md py-space-sm flex items-center justify-between bg-surface-container-high">
            <div className="flex items-center gap-space-sm">
              <span className="w-2 h-2 rounded-full bg-outline-variant"></span>
              <span className="font-label-md text-label-md uppercase tracking-wider font-semibold text-on-surface">Original Photo</span>
            </div>
          </div>
          <div className="relative w-full aspect-[4/5] bg-surface-container-lowest flex items-center justify-center p-space-sm overflow-hidden group">
            {webcamActive ? (
              <div className="relative w-full h-full flex flex-col items-center justify-center">
                <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover rounded-lg scale-x-[-1]" />
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-space-md">
                  <button onClick={captureWebcam} className="bg-primary text-on-primary rounded-full p-4 shadow-lg hover:bg-primary-container hover:text-on-primary-container transition-all flex items-center justify-center">
                    <span className="material-symbols-outlined text-[28px]">camera</span>
                  </button>
                  <button onClick={stopWebcam} className="bg-error text-on-error rounded-full p-2 shadow-lg hover:bg-red-400 transition-all flex items-center justify-center">
                    <span className="material-symbols-outlined text-[20px]">close</span>
                  </button>
                </div>
                <canvas ref={canvasRef} className="hidden" />
              </div>
            ) : preview ? (
              <img className="w-full h-full object-cover rounded-lg" src={preview} alt="Preview" />
            ) : <span className="text-on-surface-variant">Awaiting Upload or Capture</span>}
          </div>
        </div>
        <div className="flex flex-col bg-surface-container rounded-xl shadow-lg overflow-hidden border border-surface-container-high">
          <div className="px-space-md py-space-sm flex items-center justify-between bg-surface-container-high">
            <div className="flex items-center gap-space-sm">
              <span className="w-2 h-2 rounded-full bg-secondary"></span>
              <span className="font-label-md text-label-md uppercase tracking-wider font-semibold text-on-surface">Generated Sketch</span>
            </div>
            {result && (
              <button onClick={() => {
                  const a = document.createElement('a');
                  a.href = result.output_image;
                  a.download = `synthesized_sketch_${Date.now()}.png`;
                  a.click();
              }} className="flex items-center gap-space-xs text-primary hover:text-secondary transition-colors cursor-pointer">
                <span className="material-symbols-outlined text-[18px]">download</span>
                <span className="font-label-sm text-label-sm font-semibold">Save</span>
              </button>
            )}
          </div>
          <div className="relative w-full aspect-[4/5] bg-surface-container-lowest flex items-center justify-center p-space-sm overflow-hidden">
            {result && <img className="w-full h-full object-cover rounded-lg" src={result.output_image} alt="Output" />}
          </div>
        </div>
      </div>

      {/* Timing and Metrics Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-lg items-stretch">
        <div className="lg:col-span-8 bg-surface-container rounded-xl shadow-md p-space-lg flex flex-col justify-center border border-surface-container-high">
            <div className="flex items-center gap-space-md">
                <div className="p-space-sm bg-surface-container-high rounded-full">
                    <span className="material-symbols-outlined text-primary text-[24px]">brush</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-on-surface font-semibold text-body-lg">GAN Generator Target: Style {styleIdx + 1}</span>
                    <span className="text-on-surface-variant text-label-md">Using optimized conditional latent injection</span>
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
