import { useEffect, useRef, useState } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Badge } from '../components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { 
  Plus, 
  Trash2, 
  Edit, 
  Save,
  MapPin,
  AlertTriangle
} from 'lucide-react';

interface Zone {
  id: string;
  name: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
}

const ZONES_STORAGE_KEY = 'intentwatch.zones.v1';
const NORMALIZED_ZONES_STORAGE_KEY = 'intentwatch.zones.normalized.v1';

const DEFAULT_ZONES: Zone[] = [
  {
    id: '1',
    name: 'Restricted Area',
    severity: 'critical',
    x: 100,
    y: 100,
    width: 200,
    height: 150,
    color: '#ef4444'
  },
  {
    id: '2',
    name: 'High-Risk Zone',
    severity: 'high',
    x: 350,
    y: 150,
    width: 180,
    height: 120,
    color: '#f97316'
  },
  {
    id: '3',
    name: 'Monitoring Area',
    severity: 'medium',
    x: 150,
    y: 300,
    width: 250,
    height: 100,
    color: '#eab308'
  }
];

function loadZonesFromStorage(): Zone[] {
  try {
    const raw = window.localStorage.getItem(ZONES_STORAGE_KEY);
    if (!raw) return DEFAULT_ZONES;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return DEFAULT_ZONES;
    // Basic shape validation
    const sanitized = parsed
      .filter((z: any) => z && typeof z.id === 'string')
      .map((z: any) => ({
        id: String(z.id),
        name: String(z.name ?? 'Zone'),
        severity: (['critical', 'high', 'medium', 'low'].includes(z.severity) ? z.severity : 'low') as Zone['severity'],
        x: Number(z.x ?? 0),
        y: Number(z.y ?? 0),
        width: Number(z.width ?? 100),
        height: Number(z.height ?? 80),
        color: String(z.color ?? '#3b82f6'),
      } as Zone));
    return sanitized.length ? sanitized : DEFAULT_ZONES;
  } catch {
    return DEFAULT_ZONES;
  }
}

function saveZonesToStorage(zones: Zone[]) {
  try {
    window.localStorage.setItem(ZONES_STORAGE_KEY, JSON.stringify(zones));
  } catch {
    // ignore
  }
}

function saveNormalizedZonesToStorage(zones: Zone[], previewSize: { width: number; height: number } | null) {
  try {
    if (!previewSize || previewSize.width <= 0 || previewSize.height <= 0) return;
    const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
    const normalized = zones.map((z) => ({
      id: z.id,
      name: z.name,
      severity: z.severity,
      x: clamp01(z.x / previewSize.width),
      y: clamp01(z.y / previewSize.height),
      width: clamp01(z.width / previewSize.width),
      height: clamp01(z.height / previewSize.height),
    }));
    window.localStorage.setItem(NORMALIZED_ZONES_STORAGE_KEY, JSON.stringify(normalized));
  } catch {
    // ignore
  }
}

export function ZoneConfig() {
  const [zones, setZones] = useState<Zone[]>(() => loadZonesFromStorage());

  const [selectedZone, setSelectedZone] = useState<Zone | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  const previewRef = useRef<HTMLDivElement>(null);
  const interactionRef = useRef<null | {
    zoneId: string;
    mode: 'move' | 'resize-se';
    startClientX: number;
    startClientY: number;
    startZone: Zone;
  }>(null);

  const getSeverityColor = (severity: string) => {
    const colors = {
      critical: 'bg-red-500/10 text-red-400 border-red-500/20',
      high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
      medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
      low: 'bg-blue-500/10 text-blue-400 border-blue-500/20'
    };
    return colors[severity as keyof typeof colors] || colors.low;
  };

  const addNewZone = () => {
    const newZone: Zone = {
      id: Date.now().toString(),
      name: 'New Zone',
      severity: 'medium',
      x: 200,
      y: 200,
      width: 150,
      height: 100,
      color: '#eab308'
    };
    setZones([...zones, newZone]);
    setSelectedZone(newZone);
    setIsEditing(true);
  };

  const deleteZone = (id: string) => {
    setZones(zones.filter(z => z.id !== id));
    if (selectedZone?.id === id) {
      setSelectedZone(null);
      setIsEditing(false);
    }
  };

  const updateZone = (updatedZone: Zone) => {
    setZones((prev) => prev.map(z => z.id === updatedZone.id ? updatedZone : z));
    setSelectedZone(updatedZone);
  };

  const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

  const getPreviewSize = () => {
    const el = previewRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  };

  useEffect(() => {
    const onPointerMove = (e: PointerEvent) => {
      const interaction = interactionRef.current;
      if (!interaction) return;

      const size = getPreviewSize();
      if (!size) return;

      const dx = e.clientX - interaction.startClientX;
      const dy = e.clientY - interaction.startClientY;
      const z = interaction.startZone;

      if (interaction.mode === 'move') {
        const newX = clamp(z.x + dx, 0, size.width - z.width);
        const newY = clamp(z.y + dy, 0, size.height - z.height);
        updateZone({ ...z, x: Math.round(newX), y: Math.round(newY) });
      } else if (interaction.mode === 'resize-se') {
        const minSize = 30;
        const newWidth = clamp(z.width + dx, minSize, size.width - z.x);
        const newHeight = clamp(z.height + dy, minSize, size.height - z.y);
        updateZone({ ...z, width: Math.round(newWidth), height: Math.round(newHeight) });
      }
    };

    const onPointerUp = () => {
      if (interactionRef.current) {
        interactionRef.current = null;
      }
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, []);

  const saveChanges = () => {
    setIsEditing(false);
    saveZonesToStorage(zones);
  };

  // Persist zones so they don't reset when navigating between tabs.
  useEffect(() => {
    saveZonesToStorage(zones);
    saveNormalizedZonesToStorage(zones, getPreviewSize());
  }, [zones]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Zone Configuration</h1>
          <p className="text-muted-foreground mt-1">Define and manage surveillance zones</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={addNewZone} className="bg-purple-600 hover:bg-purple-700">
            <Plus className="w-4 h-4 mr-2" />
            Add Zone
          </Button>
          {isEditing && (
            <Button onClick={saveChanges} className="bg-green-600 hover:bg-green-700">
              <Save className="w-4 h-4 mr-2" />
              Save Changes
            </Button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-slate-800 p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">Total Zones</p>
            <p className="text-2xl font-bold text-white mt-1">{zones.length}</p>
          </div>
        </Card>
        <Card className="bg-slate-900 border-slate-800 p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">Critical Zones</p>
            <p className="text-2xl font-bold text-red-400 mt-1">
              {zones.filter(z => z.severity === 'critical').length}
            </p>
          </div>
        </Card>
        <Card className="bg-slate-900 border-slate-800 p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">High-Risk Zones</p>
            <p className="text-2xl font-bold text-orange-400 mt-1">
              {zones.filter(z => z.severity === 'high').length}
            </p>
          </div>
        </Card>
        <Card className="bg-slate-900 border-slate-800 p-4">
          <div className="text-center">
            <p className="text-slate-400 text-sm">Active Monitoring</p>
            <p className="text-2xl font-bold text-green-400 mt-1">{zones.length}</p>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Zone Preview */}
        <Card className="bg-slate-900 border-slate-800 p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-white mb-4">Zone Preview</h3>
          <div ref={previewRef} className="relative bg-slate-950 rounded-lg overflow-hidden" style={{ height: '500px' }}>
            {/* Simulated camera view background */}
            <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 opacity-50">
              <div className="grid grid-cols-8 grid-rows-6 h-full">
                {Array.from({ length: 48 }).map((_, i) => (
                  <div key={i} className="border border-slate-700/20" />
                ))}
              </div>
            </div>

            {/* Render zones */}
            {zones.map((zone) => (
              <div
                key={zone.id}
                onClick={() => {
                  setSelectedZone(zone);
                  setIsEditing(true);
                }}
                onPointerDown={(e) => {
                  // Begin drag-move interaction
                  e.stopPropagation();
                  (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
                  setSelectedZone(zone);
                  setIsEditing(true);
                  interactionRef.current = {
                    zoneId: zone.id,
                    mode: 'move',
                    startClientX: e.clientX,
                    startClientY: e.clientY,
                    startZone: zone,
                  };
                }}
                className={`absolute border-2 rounded cursor-pointer transition-all ${
                  selectedZone?.id === zone.id
                    ? 'border-white shadow-lg shadow-white/20'
                    : 'border-opacity-50 hover:border-opacity-100'
                }`}
                style={{
                  left: zone.x,
                  top: zone.y,
                  width: zone.width,
                  height: zone.height,
                  borderColor: zone.color,
                  backgroundColor: `${zone.color}20`
                }}
              >
                <div
                  className="absolute -top-7 left-0 px-2 py-1 rounded text-xs font-medium text-white whitespace-nowrap"
                  style={{ backgroundColor: zone.color }}
                >
                  {zone.name}
                </div>

                {/* Resize handle (bottom-right) shown for selected zone */}
                {selectedZone?.id === zone.id && (
                  <div
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
                      interactionRef.current = {
                        zoneId: zone.id,
                        mode: 'resize-se',
                        startClientX: e.clientX,
                        startClientY: e.clientY,
                        startZone: zone,
                      };
                    }}
                    className="absolute -bottom-2 -right-2 h-4 w-4 rounded-sm border border-white bg-slate-950 cursor-se-resize"
                    title="Resize"
                  />
                )}
              </div>
            ))}

            {zones.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <MapPin className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">No zones configured</p>
                  <p className="text-slate-500 text-sm mt-2">Click "Add Zone" to create a new zone</p>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Zone List & Editor */}
        <div className="space-y-6">
          {/* Zone Editor */}
          {selectedZone && isEditing && (
            <Card className="bg-slate-900 border-slate-800 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Edit Zone</h3>
              <div className="space-y-4">
                <div>
                  <Label className="text-slate-400">Zone Name</Label>
                  <Input
                    value={selectedZone.name}
                    onChange={(e) => updateZone({ ...selectedZone, name: e.target.value })}
                    className="mt-1 bg-slate-800 border-slate-700 text-white"
                  />
                </div>

                <div>
                  <Label className="text-slate-400">Severity Level</Label>
                  <Select
                    value={selectedZone.severity}
                    onValueChange={(value: any) => {
                      const colors = {
                        critical: '#ef4444',
                        high: '#f97316',
                        medium: '#eab308',
                        low: '#3b82f6'
                      };
                      updateZone({
                        ...selectedZone,
                        severity: value,
                        color: colors[value as keyof typeof colors]
                      });
                    }}
                  >
                    <SelectTrigger className="mt-1 bg-slate-800 border-slate-700 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="critical">Critical</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-slate-400">X Position</Label>
                    <Input
                      type="number"
                      value={selectedZone.x}
                      onChange={(e) => updateZone({ ...selectedZone, x: Number(e.target.value) })}
                      className="mt-1 bg-slate-800 border-slate-700 text-white"
                    />
                  </div>
                  <div>
                    <Label className="text-slate-400">Y Position</Label>
                    <Input
                      type="number"
                      value={selectedZone.y}
                      onChange={(e) => updateZone({ ...selectedZone, y: Number(e.target.value) })}
                      className="mt-1 bg-slate-800 border-slate-700 text-white"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-slate-400">Width</Label>
                    <Input
                      type="number"
                      value={selectedZone.width}
                      onChange={(e) => updateZone({ ...selectedZone, width: Number(e.target.value) })}
                      className="mt-1 bg-slate-800 border-slate-700 text-white"
                    />
                  </div>
                  <div>
                    <Label className="text-slate-400">Height</Label>
                    <Input
                      type="number"
                      value={selectedZone.height}
                      onChange={(e) => updateZone({ ...selectedZone, height: Number(e.target.value) })}
                      className="mt-1 bg-slate-800 border-slate-700 text-white"
                    />
                  </div>
                </div>

                <Button
                  onClick={() => deleteZone(selectedZone.id)}
                  variant="outline"
                  className="w-full border-red-500/20 text-red-400 hover:bg-red-500/10"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Zone
                </Button>
              </div>
            </Card>
          )}

          {/* Zone List */}
          <Card className="bg-slate-900 border-slate-800 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Configured Zones</h3>
            <div className="space-y-2">
              {zones.map((zone) => (
                <div
                  key={zone.id}
                  onClick={() => {
                    setSelectedZone(zone);
                    setIsEditing(true);
                  }}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    selectedZone?.id === zone.id
                      ? 'bg-purple-600/20 border border-purple-600'
                      : 'bg-slate-800/50 border border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: zone.color }}
                      />
                      <div>
                        <p className="text-white font-medium">{zone.name}</p>
                        <Badge className={`${getSeverityColor(zone.severity)} mt-1`}>
                          {zone.severity}
                        </Badge>
                      </div>
                    </div>
                    <Edit className="w-4 h-4 text-slate-400" />
                  </div>
                </div>
              ))}

              {zones.length === 0 && (
                <div className="text-center py-8">
                  <AlertTriangle className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-slate-400 text-sm">No zones configured</p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
