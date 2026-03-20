import { useState, useEffect, useRef } from 'react';
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
  Maximize2,
  X,
  Plus,
} from 'lucide-react';
import { API_BASE_URL, videoAPI, alertsAPI, Alert } from '../../services/api';
import { ensureNotificationPermissionNonBlocking } from '../../services/alertNotifications';

export function LiveFeed() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [activeSource, setActiveSource] = useState<'webcam' | 'file'>('webcam');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamNonce, setStreamNonce] = useState(0);
  const [extraStreams, setExtraStreams] = useState<Array<{ id: string; source: string }>>([]);
  const [extraStreaming, setExtraStreaming] = useState<Record<string, boolean>>({});
  const [extraStreamNonce, setExtraStreamNonce] = useState<Record<string, number>>({});
  const [newCameraSource, setNewCameraSource] = useState('');
  const [fullscreenStreamId, setFullscreenStreamId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reconnectAttemptsRef = useRef(0);
  const isMountedRef = useRef(true);
  const isStreamingRef = useRef(false);
  const extraStreamsRef = useRef<Array<{ id: string; source: string }>>([]);
  const streamUrl = `${videoAPI.getStreamUrl()}?t=${streamNonce}`;

  const resolveSnapshotUrl = (raw: string | null | undefined) => {
    const s = String(raw || '').trim();
    if (!s) return null;
    if (s.startsWith('http://') || s.startsWith('https://')) return s;
    if (s.startsWith('/')) return `${API_BASE_URL}${s}`;
    return `${API_BASE_URL}/${s}`;
  };

  useEffect(() => {
    extraStreamsRef.current = extraStreams;
  }, [extraStreams]);

  const getStreamUrlForId = (id: string) => {
    if (id === 'primary') return `${videoAPI.getStreamUrl()}?t=${streamNonce}`;
    const nonce = extraStreamNonce[id] ?? 0;
    return `${videoAPI.getStreamUrlById(id)}?t=${nonce}`;
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

  // Track mount/unmount to avoid setting state after navigation.
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Keep a ref of streaming state for unmount cleanup.
  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // IMPORTANT: Do NOT auto-stop streams on navigation.
  // The user expects streams (and metrics) to keep running while browsing other pages.

  // On mount, sync UI state with backend streams so the page reflects any already-running workers.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const listed = await videoAPI.listStreams();
        if (cancelled) return;
        const streams = Array.isArray(listed?.streams) ? listed.streams : [];

        const primary = streams.find((s: any) => String(s?.id) === 'primary');
        const primaryRunning = Boolean(primary?.running);
        setIsStreaming(primaryRunning);
        if (primaryRunning) {
          setStreamNonce((n) => n + 1);
          setActiveSource(primary?.mode === 'file' ? 'file' : 'webcam');
        }

        const extras = streams
          .filter((s: any) => String(s?.id) !== 'primary')
          .map((s: any) => ({ id: String(s.id), source: String(s?.path ?? '') }));
        setExtraStreams(extras);
        setExtraStreaming(
          extras.reduce((acc: Record<string, boolean>, s) => {
            acc[s.id] = true;
            return acc;
          }, {})
        );
        setExtraStreamNonce(
          extras.reduce((acc: Record<string, number>, s) => {
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

  // Fetch alerts periodically
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const data = await alertsAPI.getLiveAlerts();
        if (isMountedRef.current) {
          const next = Array.isArray(data) ? data.slice(-5) : [];
          setAlerts(next);
        }
      } catch (error) {
        console.error('Error fetching alerts:', error);
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 500); // Poll frequently for near-real-time alerts

    return () => clearInterval(interval);
  }, []);

  const handleStartStop = async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (isStreaming) {
        await videoAPI.stopVideo();
        if (isMountedRef.current) {
          setIsStreaming(false);
          setActiveSource('webcam');
        }
        reconnectAttemptsRef.current = 0;
      } else {
        ensureNotificationPermissionNonBlocking();
        const response = await withTimeout(videoAPI.startWebcam(0), 12000, 'Starting webcam');
        console.log('Webcam started:', response);

        // IMPORTANT: set zones AFTER starting (starting recreates the worker).
        try {
          const zones = loadNormalizedZones();
          if (zones.length) {
            await withTimeout(videoAPI.setZones(zones), 8000, 'Setting zones');
          }
        } catch {
          // ignore
        }

        if (isMountedRef.current) {
          setActiveSource('webcam');
        }
        // Force browser to reconnect to the MJPEG stream
        reconnectAttemptsRef.current = 0;
        if (isMountedRef.current) {
          setStreamNonce((n) => n + 1);
          setIsStreaming(true);
        }
      }
    } catch (error: any) {
      console.error('Error toggling stream:', error);
      if (isMountedRef.current) {
        setError(error.message || 'Failed to start/stop stream');
      }
      alert('Failed to start/stop stream. Make sure the backend is running on http://localhost:8000');
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  };

  const handleAddCamera = async () => {
    const raw = String(newCameraSource || '').trim();
    if (!raw) return;

    const safeId = raw.replace(/[^a-zA-Z0-9_-]/g, '_');
    const streamId = `cam-${safeId}`;
    if (extraStreams.some((s) => s.id === streamId)) return;

    setIsLoading(true);
    setError(null);
    try {
      ensureNotificationPermissionNonBlocking();
      await withTimeout(videoAPI.startStream(streamId, raw), 12000, 'Starting camera');

      // IMPORTANT: set zones AFTER starting (starting recreates the worker).
      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZonesForStream(streamId, zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      if (isMountedRef.current) {
        setExtraStreams((prev) => [...prev, { id: streamId, source: raw }]);
        setExtraStreaming((prev) => ({ ...prev, [streamId]: true }));
        setExtraStreamNonce((prev) => ({ ...prev, [streamId]: (prev[streamId] ?? 0) + 1 }));
      }
    } catch (e: any) {
      console.error('Error starting camera stream:', e);
      if (isMountedRef.current) {
        setError(e?.message || 'Failed to start camera stream');
      }
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  };

  const handleStopExtraStream = async (streamId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await videoAPI.stopStream(streamId);
      if (isMountedRef.current) {
        setExtraStreaming((prev) => ({ ...prev, [streamId]: false }));
      }
    } catch (e: any) {
      console.error('Error stopping camera stream:', e);
      if (isMountedRef.current) {
        setError(e?.message || 'Failed to stop camera stream');
      }
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
      // Stop any active stream first so the backend releases the current capture cleanly.
      if (isStreaming) {
        await withTimeout(videoAPI.stopVideo(), 8000, 'Stopping stream');
        if (isMountedRef.current) {
          setIsStreaming(false);
        }
      }

      const response = await withTimeout(videoAPI.uploadVideo(file), 30000, 'Uploading video');
      console.log('Video uploaded:', response);

      // Best-effort: send configured zones before starting.
      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZones(zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      // Explicitly select the uploaded video as the current source (avoids edge-case races).
      // Use the absolute path returned by backend.
      if (response?.path) {
        await withTimeout(videoAPI.startVideo(String(response.path)), 12000, 'Starting video');
      }

      // IMPORTANT: set zones AFTER starting (starting recreates the worker).
      try {
        const zones = loadNormalizedZones();
        if (zones.length) {
          await withTimeout(videoAPI.setZones(zones), 8000, 'Setting zones');
        }
      } catch {
        // ignore
      }

      // Start streaming the MJPEG endpoint.
      reconnectAttemptsRef.current = 0;
      if (isMountedRef.current) {
        setStreamNonce((n) => n + 1);
        setIsStreaming(true);
        setActiveSource('file');
      }
    } catch (error: any) {
      console.error('Error uploading video:', error);
      if (isMountedRef.current) {
        setError(error?.message || 'Failed to upload video');
      }
      alert('Failed to upload video. Make sure the backend is running.');
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
      // Allow re-uploading the same file (otherwise onChange may not fire).
      if (event.target) {
        event.target.value = '';
      }
    }
  };

  const getSeverityColor = (type: string) => {
    const t = (type || '').toLowerCase();
    if (t.includes('weapon')) return 'text-red-400';
    if (t.includes('loiter') || t.includes('bag') || t.includes('zone')) return 'text-orange-400';
    if (t.includes('running')) return 'text-blue-400';
    return 'text-blue-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Live Feed</h1>
        <p className="text-muted-foreground mt-1">Real-time video surveillance with AI detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Feed */}
        <Card className="p-6 lg:col-span-2">
          <div className="space-y-4">
            {/* Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Button
                  onClick={handleStartStop}
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
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload
                </Button>
              </div>
            </div>

            {/* Add Camera */}
            <div className="flex items-center gap-2">
              <Input
                value={newCameraSource}
                onChange={(e) => setNewCameraSource(e.target.value)}
                placeholder="Enter camera IP/URL (e.g., 10.12.26.111:8080)"
                className="max-w-xs"
              />
              <Button variant="outline" onClick={handleAddCamera} disabled={isLoading}>
                <Plus className="w-4 h-4 mr-2" />
                Add Camera
              </Button>
            </div>

            {error && (
              <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">
                {error}
              </div>
            )}

            {/* Video Display (Multi-camera grid) */}
            <div className={`grid gap-3 ${extraStreams.length ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
              {/* Primary stream tile */}
              <div className="relative aspect-video bg-slate-950 rounded-lg overflow-hidden border border-slate-700">
                {isStreaming ? (
                  <>
                    <img
                      src={streamUrl}
                      alt="Primary video stream"
                      className="w-full h-full object-contain"
                      onDoubleClick={() => setFullscreenStreamId('primary')}
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
                              if (isMountedRef.current) {
                                setStreamNonce((n) => n + 1);
                              }
                            }, 500);
                          }
                        })();
                      }}
                    />

                    <Button
                      variant="outline"
                      className="absolute top-2 right-2"
                      onClick={() => setFullscreenStreamId('primary')}
                    >
                      <Maximize2 className="w-4 h-4" />
                    </Button>

                    {/* Status overlay (webcam only) */}
                    {activeSource === 'webcam' && (
                      <div className="absolute top-4 left-4 space-y-2">
                        <Badge className="bg-red-600 text-white flex items-center gap-2">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                          LIVE
                        </Badge>
                        <div className="text-white text-sm bg-black/50 px-2 py-1 rounded backdrop-blur-sm">
                          {new Date().toLocaleTimeString()}
                        </div>
                      </div>
                    )}

                    {/* Stats overlay */}
                    <div className="absolute bottom-4 left-4 flex gap-4">
                      <div className="bg-black/50 text-white text-sm px-3 py-2 rounded backdrop-blur-sm">
                        <Activity className="w-4 h-4 inline mr-2" />
                        AI Detection Active
                      </div>
                      <div className="bg-black/50 text-white text-sm px-3 py-2 rounded backdrop-blur-sm">
                        <Camera className="w-4 h-4 inline mr-2" />
                        {activeSource === 'webcam' ? 'Webcam' : 'Video File'}
                      </div>
                    </div>
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

              {/* Extra camera tiles */}
              {extraStreams.map((s) => (
                <div key={s.id} className="relative aspect-video bg-slate-950 rounded-lg overflow-hidden border border-slate-700">
                  {extraStreaming[s.id] ? (
                    <>
                      <img
                        src={getStreamUrlForId(s.id)}
                        alt={`Stream ${s.id}`}
                        className="w-full h-full object-contain"
                        onDoubleClick={() => setFullscreenStreamId(s.id)}
                        onError={() => {
                          if (isMountedRef.current) {
                            setExtraStreaming((prev) => ({ ...prev, [s.id]: false }));
                          }
                        }}
                      />
                      <div className="absolute top-2 left-2">
                        <Badge className="bg-red-600 text-white">LIVE</Badge>
                      </div>
                      <div className="absolute top-2 right-2 flex gap-2">
                        <Button variant="outline" onClick={() => setFullscreenStreamId(s.id)}>
                          <Maximize2 className="w-4 h-4" />
                        </Button>
                        <Button variant="outline" onClick={() => handleStopExtraStream(s.id)}>
                          <Pause className="w-4 h-4" />
                        </Button>
                      </div>
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
                      <div className="text-center">
                        <Camera className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                        <p className="text-slate-400 mb-2">Camera {s.source}</p>
                        <p className="text-slate-500 text-sm">Stream inactive</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Fullscreen overlay */}
            {fullscreenStreamId && (
              <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
                <div className="relative w-full max-w-6xl">
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    <img
                      src={getStreamUrlForId(fullscreenStreamId)}
                      alt="Fullscreen stream"
                      className="w-full h-full object-contain"
                    />
                    <Button
                      variant="outline"
                      className="absolute top-3 right-3"
                      onClick={() => setFullscreenStreamId(null)}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Side Panel */}
        <div className="space-y-6">
          {/* System Status */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">System Status</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Backend</span>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                  Connected
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">AI Model</span>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                  YOLOv8
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Stream Status</span>
                <Badge className={isStreaming ? "bg-green-500/10 text-green-400 border-green-500/20" : "bg-slate-500/10 text-slate-400 border-slate-500/20"}>
                  {isStreaming ? 'Active' : 'Inactive'}
                </Badge>
              </div>
            </div>
          </Card>

          {/* Live Alerts */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">Live Alerts</h3>
              <Badge className="bg-red-500/10 text-red-400 border-red-500/20">
                {alerts.length}
              </Badge>
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {alerts.length === 0 ? (
                <div className="text-center py-8">
                  <AlertTriangle className="w-12 h-12 text-slate-600 mx-auto mb-2" />
                  <p className="text-slate-400 text-sm">No alerts yet</p>
                  <p className="text-slate-500 text-xs mt-1">Start the stream to detect events</p>
                </div>
              ) : (
                alerts.map((alert, index) => (
                  <div
                    key={index}
                    className="p-3 bg-muted/50 rounded-lg border border-border"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-2">
                        <AlertTriangle className={`w-4 h-4 mt-0.5 ${getSeverityColor(alert.type)}`} />
                        <div>
                          <p className="text-foreground text-sm font-medium">{alert.type}</p>
                          <p className="text-muted-foreground text-xs mt-1">{alert.message}</p>
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
                  </div>
                ))
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
