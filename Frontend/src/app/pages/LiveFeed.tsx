import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Input } from '../components/ui/input';
import {
  Play,
  Pause,
  Upload,
  Video,
  AlertTriangle,
  Activity,
  Camera,
  Plus,
  Trash2,
} from 'lucide-react';
import { API_BASE_URL, alertsAPI, videoAPI, Alert } from '../../services/api';
import { ensureNotificationPermissionNonBlocking } from '../../services/alertNotifications';

type ExtraCam = { id: string; source: string; name?: string };

export function LiveFeed() {
  const reduceMotion = useReducedMotion();

  const [isStreaming, setIsStreaming] = useState(false);
  const [primaryMode, setPrimaryMode] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamNonce, setStreamNonce] = useState(0);

  const [extraStreams, setExtraStreams] = useState<ExtraCam[]>([]);
  const [extraStreaming, setExtraStreaming] = useState<Record<string, boolean>>({});
  const [extraStreamNonce, setExtraStreamNonce] = useState<Record<string, number>>({});

  const [newCameraSource, setNewCameraSource] = useState('');
  const [newCameraName, setNewCameraName] = useState('');

  const [selectedStreamId, setSelectedStreamId] = useState<string | null>(null);
  const [clockNow, setClockNow] = useState<number>(() => Date.now());

  const fileInputRef = useRef<HTMLInputElement>(null);
  const reconnectAttemptsRef = useRef(0);
  const isMountedRef = useRef(true);
  const extraStreamsRef = useRef<ExtraCam[]>([]);

  const streamUrl = `${videoAPI.getStreamUrl()}?t=${streamNonce}`;

  const CAMERA_STORAGE_KEY = 'intentwatch.cameras.v1';

  const resolveSnapshotUrl = (raw: string | null | undefined) => {
    const s = String(raw || '').trim();
    if (!s) return null;
    if (s.startsWith('http://') || s.startsWith('https://')) return s;
    if (s.startsWith('/')) return `${API_BASE_URL}${s}`;
    return `${API_BASE_URL}/${s}`;
  };

  const loadSavedCamerasById = (): Record<string, ExtraCam> => {
    try {
      const raw = window.localStorage.getItem(CAMERA_STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return {};
      const byId: Record<string, ExtraCam> = {};
      for (const item of parsed) {
        if (!item || typeof item.id !== 'string') continue;
        const id = String(item.id);
        const source = String(item.source ?? '');
        const name = item.name != null ? String(item.name) : undefined;
        if (!id || id === 'primary' || !source) continue;
        byId[id] = { id, source, name };
      }
      return byId;
    } catch {
      return {};
    }
  };

  const upsertSavedCamera = (cam: ExtraCam) => {
    try {
      const byId = loadSavedCamerasById();
      byId[cam.id] = { id: cam.id, source: cam.source, name: cam.name };
      window.localStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify(Object.values(byId)));
    } catch {
      // ignore
    }
  };

  const removeSavedCamera = (cameraId: string) => {
    const id = String(cameraId || '').trim();
    if (!id || id === 'primary') return;
    try {
      const byId = loadSavedCamerasById();
      delete byId[id];
      window.localStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify(Object.values(byId)));
    } catch {
      // ignore
    }
  };

  const getCameraDisplayName = (cam: { id: string; source: string; name?: string } | null | undefined) => {
    if (!cam) return 'Camera';
    const n = String(cam.name ?? '').trim();
    return n || cam.source || cam.id;
  };

  const getSavedCameraNameForAlert = (cameraId: string | null | undefined) => {
    const id = String(cameraId || '').trim();
    if (!id) return null;
    if (id === 'primary') return 'Primary';
    const saved = loadSavedCamerasById()[id];
    if (!saved) return null;
    const n = String(saved.name ?? '').trim();
    return n || null;
  };

  const getStreamUrlForId = (id: string) => {
    if (id === 'primary') return `${videoAPI.getStreamUrl()}?t=${streamNonce}`;
    const nonce = extraStreamNonce[id] ?? 0;
    return `${videoAPI.getStreamUrlById(id)}?t=${nonce}`;
  };

  const getStreamRunningById = (id: string) => {
    if (id === 'primary') return isStreaming;
    return Boolean(extraStreaming[id]);
  };

  const loadNormalizedZones = () => {
    try {
      const raw = window.localStorage.getItem('intentwatch.zones.normalized.v1');
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((z: any) => z && typeof z.id === 'string')
        .map((z: any) => ({
          id: String(z.id),
          name: String(z.name ?? 'Zone'),
          severity: String(z.severity ?? 'medium'),
          x: Number(z.x ?? 0),
          y: Number(z.y ?? 0),
          width: Number(z.width ?? 0.2),
          height: Number(z.height ?? 0.2),
        }));
    } catch {
      return [];
    }
  };

  useEffect(() => {
    extraStreamsRef.current = extraStreams;
  }, [extraStreams]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const withTimeout = async <T,>(promise: Promise<T>, ms: number, label: string): Promise<T> => {
    let timeoutHandle: number | undefined;
    const timeoutPromise = new Promise<T>((_, reject) => {
      timeoutHandle = window.setTimeout(() => reject(new Error(`${label} timed out`)), ms);
    });
    try {
      return await Promise.race([promise, timeoutPromise]);
    } finally {
      if (timeoutHandle !== undefined) window.clearTimeout(timeoutHandle);
    }
  };

  // Sync UI state with backend streams and merge with saved cameras
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const listed = await videoAPI.listStreams();
        if (cancelled) return;
        const streams = Array.isArray(listed?.streams) ? listed.streams : [];

        const primary = streams.find((s: any) => String(s?.id) === 'primary');
        const primaryRunning = Boolean(primary?.running);
        const nextPrimaryMode = primary?.mode != null ? String(primary.mode) : null;
        setIsStreaming(primaryRunning);
        setPrimaryMode(nextPrimaryMode);
        if (primaryRunning) {
          setStreamNonce((n) => n + 1);
        }

        const savedById = loadSavedCamerasById();

        const runningExtras: ExtraCam[] = streams
          .filter((s: any) => String(s?.id) !== 'primary')
          .map((s: any) => {
            const id = String(s.id);
            const source = String(s?.path ?? '');
            const saved = savedById[id];
            return { id, source, name: saved?.name };
          });

        const mergedExtras = [...runningExtras];
        for (const saved of Object.values(savedById)) {
          if (mergedExtras.some((x) => x.id === saved.id)) continue;
          mergedExtras.push(saved);
        }

        setExtraStreams(mergedExtras);
        setExtraStreaming(
          mergedExtras.reduce((acc: Record<string, boolean>, s) => {
            acc[s.id] = runningExtras.some((r) => r.id === s.id);
            return acc;
          }, {})
        );
        setExtraStreamNonce(
          mergedExtras.reduce((acc: Record<string, number>, s) => {
            acc[s.id] = (acc[s.id] ?? 0) + 1;
            return acc;
          }, {})
        );
      } catch {
        // ignore
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch alerts periodically
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const data = await alertsAPI.getLiveAlerts();
        if (isMountedRef.current) {
          const next = Array.isArray(data) ? data.slice(-50) : [];
          setAlerts(next);
        }
      } catch {
        // ignore
      }
    };

    fetchAlerts();
    const intervalMs = isStreaming ? 1000 : 2500;
    const interval = window.setInterval(fetchAlerts, intervalMs);
    return () => window.clearInterval(interval);
  }, [isStreaming]);

  const handleStartStopPrimary = async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (isStreaming) {
        await videoAPI.stopVideo();
        if (isMountedRef.current) {
          setIsStreaming(false);
          setPrimaryMode(null);
        }
        reconnectAttemptsRef.current = 0;
        return;
      }

      ensureNotificationPermissionNonBlocking();
      const response = await withTimeout(videoAPI.startWebcam(0), 12000, 'Starting webcam');
      console.log('Webcam started:', response);

      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZones(zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      reconnectAttemptsRef.current = 0;
      if (isMountedRef.current) {
        setStreamNonce((n) => n + 1);
        setIsStreaming(true);
        setPrimaryMode('camera');
      }
    } catch (e: any) {
      const msg = String(e?.message || 'Failed to start/stop stream');
      if (isMountedRef.current) {
        setError(msg);
      }
      alert(msg);
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);
    try {
      ensureNotificationPermissionNonBlocking();
      if (isStreaming) {
        await withTimeout(videoAPI.stopVideo(), 8000, 'Stopping stream');
        if (isMountedRef.current) setIsStreaming(false);
      }

      const response = await withTimeout(videoAPI.uploadVideo(file), 30000, 'Uploading video');
      console.log('Video uploaded:', response);

      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZones(zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      if (response?.path) {
        await withTimeout(videoAPI.startVideo(String(response.path)), 12000, 'Starting video');
      }

      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZones(zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      reconnectAttemptsRef.current = 0;
      if (isMountedRef.current) {
        setStreamNonce((n) => n + 1);
        setIsStreaming(true);
        setPrimaryMode('file');
        setSelectedStreamId('primary');
      }
    } catch (e: any) {
      if (isMountedRef.current) setError(e?.message || 'Failed to upload video');
      alert('Failed to upload video. Make sure the backend is running.');
    } finally {
      if (isMountedRef.current) setIsLoading(false);
      if (event.target) event.target.value = '';
    }
  };

  const handleAddCamera = async () => {
    const raw = String(newCameraSource || '').trim();
    if (!raw) return;

    const rawName = String(newCameraName || '').trim();
    const idBase = (rawName || raw).replace(/[^a-zA-Z0-9_-]/g, '_');
    let streamId = `cam-${idBase}`;
    if (extraStreams.some((s) => s.id === streamId)) {
      streamId = `${streamId}-${Math.random().toString(16).slice(2, 6)}`;
    }
    if (extraStreams.some((s) => s.id === streamId)) return;

    setIsLoading(true);
    setError(null);
    try {
      ensureNotificationPermissionNonBlocking();
      await withTimeout(videoAPI.startStream(streamId, raw), 12000, 'Starting camera');

      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZonesForStream(streamId, zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      const cam: ExtraCam = { id: streamId, source: raw, name: rawName || undefined };
      upsertSavedCamera(cam);

      if (isMountedRef.current) {
        setExtraStreams((prev) => [...prev, cam]);
        setExtraStreaming((prev) => ({ ...prev, [streamId]: true }));
        setExtraStreamNonce((prev) => ({ ...prev, [streamId]: (prev[streamId] ?? 0) + 1 }));
        setNewCameraSource('');
        setNewCameraName('');
      }
    } catch (e: any) {
      if (isMountedRef.current) setError(e?.message || 'Failed to start camera stream');
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  };

  const handleStopExtraStream = async (streamId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await videoAPI.stopStream(streamId);
      if (isMountedRef.current) setExtraStreaming((prev) => ({ ...prev, [streamId]: false }));
    } catch (e: any) {
      if (isMountedRef.current) setError(e?.message || 'Failed to stop camera stream');
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  };

  const handleStartExtraStream = async (streamId: string) => {
    const cam = extraStreamsRef.current.find((s) => s.id === streamId);
    if (!cam) return;

    setIsLoading(true);
    setError(null);
    try {
      ensureNotificationPermissionNonBlocking();
      await withTimeout(videoAPI.startStream(streamId, cam.source), 12000, 'Starting camera');

      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZonesForStream(streamId, zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      if (isMountedRef.current) {
        setExtraStreaming((prev) => ({ ...prev, [streamId]: true }));
        setExtraStreamNonce((prev) => ({ ...prev, [streamId]: (prev[streamId] ?? 0) + 1 }));
      }
    } catch (e: any) {
      if (isMountedRef.current) setError(e?.message || 'Failed to start camera stream');
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  };

  const handleDeleteCamera = async (streamId: string) => {
    const id = String(streamId || '').trim();
    if (!id || id === 'primary') return;

    setIsLoading(true);
    setError(null);
    try {
      if (extraStreaming[id]) {
        try {
          await withTimeout(videoAPI.stopStream(id), 8000, 'Stopping camera');
        } catch {
          // ignore stop errors; deletion should still proceed
        }
      }

      removeSavedCamera(id);

      if (isMountedRef.current) {
        setExtraStreams((prev) => prev.filter((c) => c.id !== id));
        setExtraStreaming((prev) => {
          const next = { ...prev };
          delete next[id];
          return next;
        });
        setExtraStreamNonce((prev) => {
          const next = { ...prev };
          delete next[id];
          return next;
        });
        setSelectedStreamId((cur) => (cur === id ? null : cur));
      }
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  };

  const getSeverityColor = (type: string) => {
    const t = (type || '').toLowerCase();
    if (t.includes('weapon')) return 'text-red-400';
    if (t.includes('loiter') || t.includes('bag') || t.includes('zone')) return 'text-orange-400';
    if (t.includes('running')) return 'text-blue-400';
    return 'text-blue-400';
  };

  const tileCount = 1 + extraStreams.length;

  const computeGridColumns = (count: number) => {
    if (count <= 1) return 1;
    if (count <= 4) return 2;
    if (count <= 9) return 3;
    return 4;
  };

  const gridColumns = computeGridColumns(tileCount);

  const renderPrimaryTile = () => {
    const isFile = primaryMode === 'file';
    return (
      <div
        className="relative aspect-video bg-black overflow-hidden cursor-pointer"
        onClick={() => setSelectedStreamId('primary')}
      >
        {isStreaming ? (
          <>
            <img
              src={streamUrl}
              alt="Primary video stream"
              className="w-full h-full object-contain"
              onError={() => {
                void (async () => {
                  const attempt = reconnectAttemptsRef.current + 1;
                  reconnectAttemptsRef.current = attempt;

                  try {
                    const status = await videoAPI.getStatus();
                    if (!status?.mode) {
                      if (isMountedRef.current) {
                        setError('No video source selected. Click Start (webcam) or Upload a video.');
                        setIsStreaming(false);
                      }
                      reconnectAttemptsRef.current = 0;
                      return;
                    }
                  } catch {
                    // ignore
                  }

                  if (attempt >= 6) {
                    if (isMountedRef.current) {
                      setError('Stream unavailable. Try Stop then Start, or upload again.');
                      setIsStreaming(false);
                    }
                    reconnectAttemptsRef.current = 0;
                    return;
                  }

                  if (isMountedRef.current) {
                    setError('Stream connection lost. Reconnecting...');
                    window.setTimeout(() => {
                      if (isMountedRef.current) setStreamNonce((n) => n + 1);
                    }, 500);
                  }
                })();
              }}
            />

            {!isFile && (
              <div className="absolute top-2 left-2">
                <Badge className="bg-red-600 text-white">LIVE</Badge>
              </div>
            )}
            <div className="absolute bottom-2 left-2">
              <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">Primary</div>
            </div>

            {!isFile && (
              <div className="absolute bottom-2 right-2 flex gap-2">
                <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                  <Activity className="w-3 h-3 inline mr-1" />
                  AI Active
                </div>
                <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                  {new Date(clockNow).toLocaleTimeString()}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
            <div className="text-center">
              <Video className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400 mb-2">Primary stream stopped</p>
              <p className="text-slate-500 text-sm">Click Start to begin detection</p>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderExtraTile = (cam: ExtraCam) => {
    const running = Boolean(extraStreaming[cam.id]);
    return (
      <div
        key={cam.id}
        className="relative aspect-video bg-black overflow-hidden cursor-pointer"
        onClick={() => setSelectedStreamId(cam.id)}
      >
        <div className="absolute top-2 right-2 z-10 flex gap-2" onClick={(e) => e.stopPropagation()}>
          {running && (
            <Button variant="outline" onClick={() => handleStopExtraStream(cam.id)}>
              <Pause className="w-4 h-4" />
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => handleDeleteCamera(cam.id)}
            disabled={isLoading}
            title="Delete camera"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>

        {running ? (
          <>
            <img
              src={getStreamUrlForId(cam.id)}
              alt={`Stream ${cam.id}`}
              className="w-full h-full object-contain"
              onError={() => {
                if (isMountedRef.current) {
                  setExtraStreaming((prev) => ({ ...prev, [cam.id]: false }));
                  setError('Stream unavailable. Try Start again.');
                }
              }}
            />

            <div className="absolute top-2 left-2">
              <Badge className="bg-red-600 text-white">LIVE</Badge>
            </div>
            <div className="absolute bottom-2 left-2">
              <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                {getCameraDisplayName(cam)}
              </div>
            </div>
            <div className="absolute bottom-2 right-2 flex gap-2">
              <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                <Activity className="w-3 h-3 inline mr-1" />
                AI Active
              </div>
              <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                {new Date(clockNow).toLocaleTimeString()}
              </div>
            </div>
          </>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
            <div className="text-center">
              <Camera className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400 mb-1">{getCameraDisplayName(cam)}</p>
              {cam.source && <p className="text-slate-500 text-xs mb-2">{cam.source}</p>}
              <p className="text-slate-500 text-sm">Stream inactive</p>
              <div className="mt-3 flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
                <Button variant="outline" onClick={() => handleStartExtraStream(cam.id)} disabled={isLoading}>
                  <Play className="w-4 h-4 mr-2" />
                  Start
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderSelectedView = (id: string) => {
    const isPrimary = id === 'primary';
    const running = getStreamRunningById(id);
    const cam = isPrimary ? null : extraStreamsRef.current.find((s) => s.id === id);
    const name = isPrimary ? 'Primary' : getCameraDisplayName(cam);
    const isFile = isPrimary && primaryMode === 'file';

    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {!isFile && (
              <Badge className={running ? 'bg-red-600 text-white' : 'bg-slate-500/10 text-slate-300 border-slate-500/20'}>
                {running ? 'LIVE' : 'STOPPED'}
              </Badge>
            )}
            <div className="text-sm text-foreground font-medium">{name}</div>
          </div>
          {!isFile && <div className="text-xs text-muted-foreground">Double click video to go back</div>}
        </div>

        <div
          className="relative aspect-video bg-black overflow-hidden border border-slate-700"
          onDoubleClick={() => setSelectedStreamId(null)}
        >
          {running ? (
            <img src={getStreamUrlForId(id)} alt="Selected stream" className="w-full h-full object-contain" />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
              <div className="text-center">
                <Camera className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-300 mb-1">{name}</p>
                <p className="text-slate-500 text-sm mb-3">Stream inactive</p>
                {!isPrimary && (
                  <Button variant="outline" onClick={() => handleStartExtraStream(id)} disabled={isLoading}>
                    <Play className="w-4 h-4 mr-2" />
                    Start
                  </Button>
                )}
              </div>
            </div>
          )}

          <div className="absolute bottom-2 right-2 flex gap-2">
            {!isFile && (
              <>
                <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                  <Activity className="w-3 h-3 inline mr-1" />
                  AI Active
                </div>
                <div className="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                  {new Date(clockNow).toLocaleTimeString()}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Live Feed</h1>
        <p className="text-muted-foreground mt-1">Real-time video surveillance with AI detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="p-6 lg:col-span-2">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Button
                  onClick={handleStartStopPrimary}
                  disabled={isLoading}
                  className={isStreaming ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
                >
                  {isLoading ? (
                    <>Loading...</>
                  ) : isStreaming ? (
                    <>
                      <Pause className="w-4 h-4 mr-2" />
                      Stop
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start
                    </>
                  )}
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileUpload} className="hidden" />
                <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Input
                value={newCameraSource}
                onChange={(e) => setNewCameraSource(e.target.value)}
                placeholder="Enter camera IP/URL (e.g., 10.12.26.111:8080)"
                className="max-w-xs"
              />
              <Input
                value={newCameraName}
                onChange={(e) => setNewCameraName(e.target.value)}
                placeholder="Name (e.g., Kitchen)"
                className="max-w-[180px]"
              />
              <Button variant="outline" onClick={handleAddCamera} disabled={isLoading}>
                <Plus className="w-4 h-4 mr-2" />
                Add Camera
              </Button>
            </div>

            {error && (
              <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">{error}</div>
            )}

            <div className="overflow-auto">
              {selectedStreamId ? (
                renderSelectedView(selectedStreamId)
              ) : primaryMode === 'file' && isStreaming ? (
                renderSelectedView('primary')
              ) : (
                <div className="bg-black border border-slate-700">
                  <div
                    className="grid gap-px bg-slate-700"
                    style={{ gridTemplateColumns: `repeat(${gridColumns}, minmax(0, 1fr))` }}
                  >
                    {renderPrimaryTile()}
                    {extraStreams.map((c) => renderExtraTile(c))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>

        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">System Status</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Backend</span>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20">Connected</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">AI Model</span>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20">YOLOv8</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Primary Stream</span>
                <Badge
                  className={
                    isStreaming
                      ? 'bg-green-500/10 text-green-400 border-green-500/20'
                      : 'bg-slate-500/10 text-slate-400 border-slate-500/20'
                  }
                >
                  {isStreaming ? 'Active' : 'Inactive'}
                </Badge>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">Live Alerts</h3>
              <Badge className="bg-red-500/10 text-red-400 border-red-500/20">{alerts.length}</Badge>
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {alerts.length === 0 ? (
                <div className="text-center py-8">
                  <AlertTriangle className="w-12 h-12 text-slate-600 mx-auto mb-2" />
                  <p className="text-slate-400 text-sm">No alerts yet</p>
                  <p className="text-slate-500 text-xs mt-1">Start the stream to detect events</p>
                </div>
              ) : (
                <AnimatePresence initial={false}>
                  {alerts.map((alert, index) => (
                    <motion.div
                      key={index}
                      initial={reduceMotion ? false : { opacity: 0, y: 6, scale: 0.98 }}
                      animate={reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 }}
                      exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -6, scale: 0.98 }}
                      transition={reduceMotion ? { duration: 0 } : { duration: 0.18, ease: 'easeOut' }}
                      className="p-3 bg-muted/50 rounded-lg border border-border"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-2">
                          <AlertTriangle className={`w-4 h-4 mt-0.5 ${getSeverityColor(alert.type)}`} />
                          <div>
                            <p className="text-foreground text-sm font-medium">{alert.type}</p>
                            <p className="text-muted-foreground text-xs mt-1">{alert.message}</p>
                            {getSavedCameraNameForAlert(alert.camera) && (
                              <p className="text-muted-foreground text-xs mt-1">
                                Camera: {getSavedCameraNameForAlert(alert.camera)}
                              </p>
                            )}
                            <p className="text-muted-foreground text-xs mt-1">{alert.time}</p>
                          </div>
                        </div>
                        {resolveSnapshotUrl(alert.snapshot_url) && (
                          <a
                            href={resolveSnapshotUrl(alert.snapshot_url) as string}
                            target="_blank"
                            rel="noreferrer"
                            className="shrink-0"
                            title="Open snapshot"
                          >
                            <img
                              src={resolveSnapshotUrl(alert.snapshot_url) as string}
                              alt="Alert snapshot"
                              className="w-16 h-16 rounded-md object-cover border border-border"
                              loading="lazy"
                            />
                          </a>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
