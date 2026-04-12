import { Card } from '../components/ui/card';
import { 
  Users, 
  AlertTriangle, 
  Video, 
  TrendingUp,
  Clock,
  Activity
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { mockStats, generateActivityData, generateAlertDistribution, generateMockAlerts } from '../data/mockData';
import { Badge } from '../components/ui/badge';
import { aiAPI, alertsAPI, AnalyticsData, API_BASE_URL, demoAPI, DemoDetectImageResponse, systemAPI } from '../../services/api';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';

const StatCard = ({ icon: Icon, label, value, trend, color }: any) => (
  <Card className="p-6">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-muted-foreground text-sm">{label}</p>
        <p className="text-3xl font-bold text-foreground mt-2">{value}</p>
        {trend && (
          <div className="flex items-center gap-1 mt-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-sm text-green-400">{trend}</span>
          </div>
        )}
      </div>
      <div className={`p-4 rounded-lg ${color}`}>
        <Icon className="w-6 h-6 text-white" />
      </div>
    </div>
  </Card>
);

export function Dashboard() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [hasLoadedAnalytics, setHasLoadedAnalytics] = useState(false);
  const [metrics, setMetrics] = useState<any | null>(null);

  const [demoFile, setDemoFile] = useState<File | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [demoResult, setDemoResult] = useState<DemoDetectImageResponse | null>(null);

  const demoFileInputRef = useRef<HTMLInputElement | null>(null);

  const [askQuestion, setAskQuestion] = useState('');
  const [askAnswer, setAskAnswer] = useState<string>('');
  const [askSources, setAskSources] = useState<any[]>([]);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  const resolveSnapshotUrl = (raw: string | null | undefined) => {
    const s = String(raw || '').trim();
    if (!s) return null;
    if (s.startsWith('http://') || s.startsWith('https://')) return s;
    if (s.startsWith('/')) return `${API_BASE_URL}${s}`;
    return `${API_BASE_URL}/${s}`;
  };

  useEffect(() => {
    let isMounted = true;
    const refresh = async () => {
      try {
        const data = await alertsAPI.getAnalytics();
        if (!isMounted) return;
        setAnalytics(data);
      } catch {
        // ignore (dashboard can still render mock data)
      } finally {
        if (isMounted) setHasLoadedAnalytics(true);
      }
    };

    refresh();
    const interval = window.setInterval(refresh, 5000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    const refresh = async () => {
      try {
        const data = await systemAPI.getMetrics();
        if (!isMounted) return;
        setMetrics(data);
      } catch {
        // ignore
      }
    };

    refresh();
    const interval = window.setInterval(refresh, 2000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const activityData = useMemo(() => {
    const base = generateActivityData();
    const byHour = analytics?.by_hour;
    if (!byHour || byHour.length === 0) return base;
    const map = new Map(byHour.map((h) => [h.hour, h.alerts] as const));
    return base.map((row) => ({ ...row, alerts: map.get(row.hour) ?? row.alerts }));
  }, [analytics?.by_hour]);

  const alertDistribution = useMemo(() => {
    const sev = analytics?.severity;
    if (!sev) return generateAlertDistribution();
    return [
      { name: 'Critical', value: sev['Critical'] ?? 0, color: '#ef4444' },
      { name: 'High', value: sev['High'] ?? 0, color: '#f97316' },
      { name: 'Medium', value: sev['Medium'] ?? 0, color: '#eab308' },
      { name: 'Low', value: sev['Low'] ?? 0, color: '#3b82f6' },
    ];
  }, [analytics?.severity]);

  const pieData = useMemo(() => {
    return alertDistribution.filter((d) => (d?.value ?? 0) > 0);
  }, [alertDistribution]);

  const recentAlerts = useMemo(() => {
    const backend = analytics?.recent;
    if (!backend || backend.length === 0) return generateMockAlerts(5);

    const normalizeBackendSeverity = (raw: any): 'critical' | 'high' | 'medium' | 'low' | null => {
      const s = String(raw ?? '').trim().toLowerCase();
      if (!s) return null;
      if (s === 'critical') return 'critical';
      if (s === 'high') return 'high';
      if (s === 'medium') return 'medium';
      if (s === 'low') return 'low';
      return null;
    };

    const severityFromType = (t: string): 'critical' | 'high' | 'medium' | 'low' => {
      const s = (t || '').toLowerCase();
      if (s.includes('weapon')) return 'critical';
      if (s.includes('bag')) return 'high';
      if (s.includes('zone')) return 'high';
      if (s.includes('loiter')) return 'medium';
      if (s.includes('running')) return 'low';
      return 'low';
    };

    return backend
      .slice(-5)
      .reverse()
      .map((a, idx) => ({
        id: `backend-${idx}-${a.timestamp ?? a.time}`,
        timestamp: a.timestamp ?? new Date().toISOString(),
        type: a.type,
        severity: normalizeBackendSeverity((a as any)?.severity) ?? severityFromType(a.type),
        description: a.message,
        location: '—',
        camera: '—',
      }));
  }, [analytics?.recent]);

  const getSeverityColor = (severity: string) => {
    const colors = {
      critical: 'bg-red-500/10 text-red-400 border-red-500/20',
      high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
      medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
      low: 'bg-blue-500/10 text-blue-400 border-blue-500/20'
    };
    return colors[severity as keyof typeof colors] || colors.low;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
        <p className="text-muted-foreground mt-1">Real-time surveillance monitoring</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Users}
          label="People Detected"
          value={metrics?.people_detected ?? mockStats.peopleDetected}
          trend="+12% from last hour"
          color="bg-purple-600"
        />
        <StatCard
          icon={AlertTriangle}
          label="Active Alerts"
          value={(metrics?.active_alerts ?? (hasLoadedAnalytics ? (analytics?.total ?? 0) : null)) ?? mockStats.activeAlerts}
          color="bg-red-600"
        />
        <StatCard
          icon={Video}
          label="Cameras Online"
          value={metrics?.cameras_online ?? mockStats.camerasOnline}
          color="bg-green-600"
        />
        <StatCard
          icon={Activity}
          label="System Uptime"
          value={metrics?.uptime ?? mockStats.uptime}
          color="bg-blue-600"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Timeline */}
        <Card className="p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-foreground mb-4">Activity Timeline (24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={activityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--popover)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--foreground)'
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="people" stroke="#8b5cf6" strokeWidth={2} name="People" />
              <Line type="monotone" dataKey="alerts" stroke="#ef4444" strokeWidth={2} name="Alerts" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Alert Distribution */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Alert Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Legend verticalAlign="middle" align="right" layout="vertical" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--popover)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--foreground)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Recent Alerts */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Recent Alerts</h3>
          <Badge variant="outline" className="bg-red-500/10 text-red-400 border-red-500/20">
            {recentAlerts.length} New
          </Badge>
        </div>
        <div className="space-y-3">
          {recentAlerts.map((alert) => (
            <div
              key={alert.id}
              className="flex items-center justify-between p-4 bg-muted/50 rounded-lg border border-border hover:bg-accent/50 transition-colors"
            >
              <div className="flex items-center gap-4">
                <Badge className={getSeverityColor(alert.severity)}>
                  {alert.severity.toUpperCase()}
                </Badge>
                <div>
                  <p className="text-foreground font-medium">{alert.type}</p>
                  <p className="text-sm text-muted-foreground">{alert.location} • {alert.camera}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Clock className="w-4 h-4" />
                {new Date(alert.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Demo: Upload Image (no streaming) */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Demo: Upload Image</h3>

        <div className="flex flex-col md:flex-row gap-3">
          <input
            ref={demoFileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = (e.currentTarget.files && e.currentTarget.files[0]) || null;
              // allow picking the same file again
              e.currentTarget.value = '';
              if (!f) return;

              setDemoFile(f);
              setDemoError(null);
              setDemoResult(null);

              void (async () => {
                setDemoLoading(true);
                try {
                  const res = await demoAPI.detectImage(f, { stream_id: 'demo', emit_alert: true });
                  setDemoResult(res);
                } catch (err: any) {
                  setDemoError(err?.message || 'Demo detection failed');
                } finally {
                  setDemoLoading(false);
                }
              })();
            }}
          />
          <Button
            disabled={demoLoading}
            onClick={() => {
              // If a file was already selected, let the user pick a new one.
              demoFileInputRef.current?.click();
            }}
          >
            {demoLoading ? 'Running…' : (demoFile ? 'Upload Another Image' : 'Upload Image')}
          </Button>
        </div>

        {demoError && <p className="text-sm text-red-400 mt-3">{demoError}</p>}

        {demoResult && (
          <div className="mt-4 space-y-3">
            {demoResult.snapshot_url && (
              <a
                href={resolveSnapshotUrl(demoResult.snapshot_url) || undefined}
                target="_blank"
                rel="noreferrer"
                className="block rounded border border-border overflow-hidden bg-black"
                title="Open annotated snapshot"
              >
                <img
                  src={resolveSnapshotUrl(demoResult.snapshot_url) || undefined}
                  alt="annotated"
                  className="w-full max-h-96 object-contain"
                />
              </a>
            )}

            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <p className="text-sm text-muted-foreground">Detections</p>
              {demoResult.detections?.length ? (
                <div className="mt-2 flex flex-wrap gap-2">
                  {demoResult.detections
                    .slice(0, 12)
                    .map((d, idx) => (
                      <Badge key={`${d.label}-${idx}`} variant="outline">
                        {d.label} ({(d.confidence ?? 0).toFixed(2)})
                      </Badge>
                    ))}
                </div>
              ) : (
                <p className="mt-2 text-sm text-foreground">No objects detected.</p>
              )}
              <p className="mt-3 text-xs text-muted-foreground">
                This also emits a Weapon alert so the Dashboard updates.
              </p>
            </div>
          </div>
        )}
      </Card>

      {/* Ask AI (RAG) */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Ask AI (Alerts)</h3>

        <div className="flex flex-col md:flex-row gap-3">
          <Input
            value={askQuestion}
            onChange={(e) => setAskQuestion(e.target.value)}
            placeholder="Ask about recent alerts (e.g., What happened last night?)"
          />
          <Button
            disabled={askLoading || !askQuestion.trim()}
            onClick={() => {
              void (async () => {
                setAskLoading(true);
                setAskError(null);
                try {
                  const res = await aiAPI.ask(askQuestion.trim(), { k: 5, max_alerts: 1000 });
                  setAskAnswer(String(res?.answer ?? ''));
                  setAskSources(Array.isArray((res as any)?.sources) ? (res as any).sources : []);
                } catch (e: any) {
                  setAskError(e?.message || 'Failed to ask AI');
                } finally {
                  setAskLoading(false);
                }
              })();
            }}
          >
            {askLoading ? 'Asking…' : 'Ask'}
          </Button>
        </div>

        {askError && <p className="text-sm text-red-400 mt-3">{askError}</p>}

        {askAnswer && (
          <pre className="mt-4 whitespace-pre-wrap rounded-lg border border-border bg-muted/30 p-4 text-sm text-foreground">
            {askAnswer}
          </pre>
        )}

        {askSources.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-muted-foreground mb-2">Related snapshots</p>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {askSources.map((s: any, idx: number) => {
                const url = resolveSnapshotUrl(s?.snapshot_url);
                if (!url) return null;
                return (
                  <a
                    key={`${s?.id ?? idx}`}
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="block rounded border border-border overflow-hidden bg-black"
                    title={String(s?.type ?? 'Alert')}
                  >
                    <img src={url} alt={String(s?.type ?? 'snapshot')} className="w-full h-24 object-cover" />
                  </a>
                );
              })}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
