import { useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

type Clip = {
  filename: string;
  size_bytes: number;
  mtime: number;
  public_url?: string | null;
  url: string; // backend relative url
};

export function History() {
  const [streamId] = useState('primary');
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [clips, setClips] = useState<Clip[]>([]);
  const [selectedClip, setSelectedClip] = useState<Clip | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [refreshToken, setRefreshToken] = useState(0);
  const [useSupabase, setUseSupabase] = useState(false);
  const [playbackError, setPlaybackError] = useState<string>('');

  const clipSrc = useMemo(() => {
    if (!selectedClip) return '';
    // Prefer streaming via backend for reliable playback (correct MIME + CORS).
    // If playback fails and we have a Supabase URL, we can fall back.
    if (!useSupabase && selectedClip.url) return `${API_BASE_URL}${selectedClip.url}`;
    if (selectedClip.public_url) return selectedClip.public_url;
    return '';
  }, [selectedClip, useSupabase]);

  useEffect(() => {
    let mounted = true;
    const loadDates = async () => {
      setError('');
      try {
        const res = await fetch(`${API_BASE_URL}/history/dates?stream_id=${encodeURIComponent(streamId)}`, { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to load history dates');
        const data = await res.json();
        const next = Array.isArray(data?.dates) ? (data.dates as string[]) : [];
        if (!mounted) return;
        setDates(next);
        setSelectedDate((prev) => prev || next[0] || '');
      } catch (e: any) {
        if (!mounted) return;
        setError(String(e?.message || e || 'Failed to load history dates'));
      }
    };
    void loadDates();
    return () => {
      mounted = false;
    };
  }, [streamId, refreshToken]);

  useEffect(() => {
    let mounted = true;
    const loadClips = async () => {
      if (!selectedDate) {
        setClips([]);
        setSelectedClip(null);
        return;
      }
      setLoading(true);
      setError('');
      try {
        const url = `${API_BASE_URL}/history/clips?stream_id=${encodeURIComponent(streamId)}&date=${encodeURIComponent(selectedDate)}`;
        const res = await fetch(url, { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to load clips');
        const data = await res.json();
        const next = Array.isArray(data?.clips) ? (data.clips as Clip[]) : [];
        if (!mounted) return;
        setClips(next);
        setSelectedClip(next[0] || null);
        setUseSupabase(false);
        setPlaybackError('');
      } catch (e: any) {
        if (!mounted) return;
        setError(String(e?.message || e || 'Failed to load clips'));
        setClips([]);
        setSelectedClip(null);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    void loadClips();
    return () => {
      mounted = false;
    };
  }, [selectedDate, streamId, refreshToken]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>History</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2">
              <div className="text-sm text-muted-foreground">Date</div>
              <Input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="w-[180px]"
              />
              {dates.length > 0 && (
                <div className="text-xs text-muted-foreground">({dates.length} days)</div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                onClick={() => {
                  setRefreshToken((n) => n + 1);
                }}
                disabled={loading}
              >
                Refresh
              </Button>
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-400">{error}</div>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <div className="text-sm font-medium mb-2">Clips</div>
              <div className="space-y-2 max-h-[420px] overflow-auto pr-1">
                {clips.length === 0 && (
                  <div className="text-sm text-muted-foreground">No clips for this date.</div>
                )}
                {clips.map((c) => {
                  const active = selectedClip?.filename === c.filename;
                  const sizeMb = (c.size_bytes / (1024 * 1024)).toFixed(1);
                  const time = new Date(c.mtime * 1000).toLocaleTimeString();
                  return (
                    <button
                      key={c.filename}
                      className={`w-full text-left rounded-md border px-3 py-2 transition-colors ${
                        active
                          ? 'bg-accent text-foreground border-border'
                          : 'bg-background hover:bg-accent/50 text-foreground border-border'
                      }`}
                      onClick={() => setSelectedClip(c)}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-medium text-sm truncate">{c.filename}</div>
                        <div className="text-xs text-muted-foreground shrink-0">{sizeMb} MB</div>
                      </div>
                      <div className="text-xs text-muted-foreground">{time}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="lg:col-span-2">
              <div className="text-sm font-medium mb-2">Playback</div>
              {selectedClip ? (
                <div className="rounded-lg border border-border bg-black/20 p-3">
                  <video
                    key={clipSrc}
                    controls
                    preload="metadata"
                    className="w-full rounded-md bg-black"
                    onError={() => {
                      // If backend playback fails (codec/range), try Supabase URL when available.
                      if (!useSupabase && selectedClip.public_url) {
                        setUseSupabase(true);
                        setPlaybackError('');
                        return;
                      }
                      setPlaybackError('This clip could not be played in your browser. Try recording a new clip after restarting the backend.');
                    }}
                  >
                    <source src={clipSrc} type="video/mp4" />
                  </video>
                  <div className="mt-2 text-xs text-muted-foreground">
                    Source: {useSupabase ? 'Supabase' : 'Local backend'}{selectedClip.public_url ? ' (uploaded)' : ''}
                  </div>
                  {playbackError && (
                    <div className="mt-2 text-xs text-red-400">{playbackError}</div>
                  )}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">Select a clip to play.</div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
