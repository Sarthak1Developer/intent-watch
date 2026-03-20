import { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { 
  Search, 
  Filter,
  Clock,
  AlertTriangle,
  Trash2
} from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { API_BASE_URL, alertsAPI, Alert } from '../../services/api';

export function AlertsLog() {
  const [searchTerm, setSearchTerm] = useState('');
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [typeFilter, setTypeFilter] = useState('all');
  const [error, setError] = useState<string | null>(null);

  const resolveSnapshotUrl = (raw: string | null | undefined) => {
    const s = String(raw || '').trim();
    if (!s) return null;
    if (s.startsWith('http://') || s.startsWith('https://')) return s;
    if (s.startsWith('/')) return `${API_BASE_URL}${s}`;
    return `${API_BASE_URL}/${s}`;
  };

  // Fetch alerts periodically
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const data = await alertsAPI.getLiveAlerts();
        setAlerts(Array.isArray(data) ? data : []);
        setError(null);
      } catch (error) {
        console.error('Error fetching alerts:', error);
        setError('Failed to connect to backend');
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, []);

  const filteredAlerts = alerts.filter(alert => {
    const matchesSearch = 
      alert.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      alert.message.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesType = typeFilter === 'all' || alert.type === typeFilter;

    return matchesSearch && matchesType;
  });

  const getSeverityColor = (type: string) => {
    if (type.includes('Loiter') || type.includes('Bag')) return 'bg-red-500/10 text-red-400 border-red-500/20';
    if (type.includes('Running')) return 'bg-orange-500/10 text-orange-400 border-orange-500/20';
    return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
  };

  const handleClearAlerts = async () => {
    try {
      await alertsAPI.clearAlerts();
      setAlerts([]);
    } catch (error) {
      console.error('Error clearing alerts:', error);
      alert('Failed to clear alerts');
    }
  };

  const alertTypes = [...new Set(alerts.map(a => a.type))];

  return (
    <div className="space-y-6">
      {/* Error Banner */}
      {error && (
        <Card className="bg-red-900/20 border-red-500/50 p-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <div>
              <p className="text-red-400 font-semibold">Connection Error</p>
              <p className="text-red-300 text-sm">{error}. Please make sure the backend server is running on http://localhost:8000</p>
            </div>
          </div>
        </Card>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Alerts Log</h1>
          <p className="text-muted-foreground mt-1">Real-time alert history and management</p>
        </div>
        <Button onClick={handleClearAlerts} variant="destructive" className="bg-red-600 hover:bg-red-700">
          <Trash2 className="w-4 h-4 mr-2" />
          Clear All Alerts
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">Total Alerts</p>
            <p className="text-2xl font-bold text-foreground mt-1">{alerts.length}</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">Alert Types</p>
            <p className="text-2xl font-bold text-purple-400 mt-1">{alertTypes.length}</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">Filtered Results</p>
            <p className="text-2xl font-bold text-blue-400 mt-1">{filteredAlerts.length}</p>
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card className="p-6">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex-1 min-w-[300px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <Input
                placeholder="Search alerts..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-48">
              <Filter className="w-4 h-4 mr-2" />
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              {alertTypes.map(type => (
                <SelectItem key={type} value={type}>{type}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* Alerts Table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-muted/50">
              <TableHead className="text-slate-400">Time</TableHead>
              <TableHead className="text-slate-400">Type</TableHead>
              <TableHead className="text-slate-400">Snapshot</TableHead>
              <TableHead className="text-slate-400">Message</TableHead>
              <TableHead className="text-slate-400 text-right">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredAlerts.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center py-12">
                  <div className="flex flex-col items-center justify-center">
                    <AlertTriangle className="w-12 h-12 text-slate-600 mb-3" />
                    <p className="text-slate-400">No alerts found</p>
                    <p className="text-slate-500 text-sm mt-1">
                      {alerts.length === 0 ? 'Start the video stream to generate alerts' : 'Try adjusting your filters'}
                    </p>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              filteredAlerts.map((alert, index) => (
                <TableRow key={index} className="hover:bg-muted/50">
                  <TableCell className="text-foreground">
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4 text-slate-500" />
                      {alert.time}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge className={getSeverityColor(alert.type)}>
                      {alert.type}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {resolveSnapshotUrl(alert.snapshot_url) ? (
                      <a
                        href={resolveSnapshotUrl(alert.snapshot_url) as string}
                        target="_blank"
                        rel="noreferrer"
                        title="Open snapshot"
                        className="inline-block"
                      >
                        <img
                          src={resolveSnapshotUrl(alert.snapshot_url) as string}
                          alt="Alert snapshot"
                          className="w-16 h-10 rounded-md object-cover border border-border"
                          loading="lazy"
                        />
                      </a>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-foreground">
                    {alert.message}
                  </TableCell>
                  <TableCell className="text-right">
                    <AlertTriangle className="w-4 h-4 text-red-400 inline" />
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      {/* Alert Count Footer */}
      {filteredAlerts.length > 0 && (
        <div className="text-center text-muted-foreground text-sm">
          Showing {filteredAlerts.length} of {alerts.length} total alerts
        </div>
      )}
    </div>
  );
}
