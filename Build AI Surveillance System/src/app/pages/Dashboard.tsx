import { Card } from '../components/ui/card';
import { 
  Users, 
  AlertTriangle, 
  Video, 
  TrendingUp,
  Clock,
  Activity
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { mockStats, generateActivityData, generateAlertDistribution, generateMockAlerts } from '../data/mockData';
import { Badge } from '../components/ui/badge';
import { alertsAPI, AnalyticsData, systemAPI } from '../../services/api';

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

    const severityFromType = (t: string): 'critical' | 'high' | 'medium' | 'low' => {
      const s = (t || '').toLowerCase();
      if (s.includes('bag')) return 'high';
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
        severity: severityFromType(a.type),
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
    </div>
  );
}
