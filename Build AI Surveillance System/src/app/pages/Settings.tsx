import { useEffect, useMemo, useState } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Slider } from '../components/ui/slider';
import { Switch } from '../components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';

type CameraSource = 'webcam' | 'video_file' | 'rtsp';
type Resolution = '640x480' | '1280x720' | '1920x1080';

type SettingsState = {
  // Camera Settings
  cameraSource: CameraSource;
  resolution: Resolution;
  fps: number;
  rtspUrl: string;

  // Detection Settings
  confidence: number;
  loiteringThreshold: number;
  bagThreshold: number;
  detectPeople: boolean;
  detectLoitering: boolean;
  detectRunning: boolean;
  detectUnattendedBag: boolean;

  // Alert Settings
  sound: boolean;
  alertLogging: boolean;
  email: boolean;
};

export function Settings() {
  const SETTINGS_STORAGE_KEY = 'intentwatch.settings.v1';

  const defaultSettings = useMemo<SettingsState>(
    () => ({
      cameraSource: 'webcam',
      resolution: '1280x720',
      fps: 30,
      rtspUrl: '',

      confidence: 50,
      loiteringThreshold: 10,
      bagThreshold: 15,
      detectPeople: true,
      detectLoitering: true,
      detectRunning: true,
      detectUnattendedBag: true,

      sound: true,
      alertLogging: true,
      email: false,
    }),
    [],
  );

  const [settings, setSettings] = useState<SettingsState>(defaultSettings);

  const loadPersistedSettings = () => {
    try {
      const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Partial<SettingsState> | null;
      if (!parsed || typeof parsed !== 'object') return;

      setSettings((prev) => {
        const next: SettingsState = { ...prev };

        const cameraSource = parsed.cameraSource;
        if (cameraSource === 'webcam' || cameraSource === 'video_file' || cameraSource === 'rtsp') next.cameraSource = cameraSource;

        const resolution = parsed.resolution;
        if (resolution === '640x480' || resolution === '1280x720' || resolution === '1920x1080') next.resolution = resolution;

        if (typeof parsed.fps === 'number' && Number.isFinite(parsed.fps)) next.fps = parsed.fps;
        if (typeof parsed.rtspUrl === 'string') next.rtspUrl = parsed.rtspUrl;

        if (typeof parsed.confidence === 'number' && Number.isFinite(parsed.confidence)) next.confidence = parsed.confidence;
        if (typeof parsed.loiteringThreshold === 'number' && Number.isFinite(parsed.loiteringThreshold)) next.loiteringThreshold = parsed.loiteringThreshold;
        if (typeof parsed.bagThreshold === 'number' && Number.isFinite(parsed.bagThreshold)) next.bagThreshold = parsed.bagThreshold;

        if (typeof parsed.detectPeople === 'boolean') next.detectPeople = parsed.detectPeople;
        if (typeof parsed.detectLoitering === 'boolean') next.detectLoitering = parsed.detectLoitering;
        if (typeof parsed.detectRunning === 'boolean') next.detectRunning = parsed.detectRunning;
        if (typeof parsed.detectUnattendedBag === 'boolean') next.detectUnattendedBag = parsed.detectUnattendedBag;

        if (typeof parsed.sound === 'boolean') next.sound = parsed.sound;
        if (typeof parsed.alertLogging === 'boolean') next.alertLogging = parsed.alertLogging;
        if (typeof parsed.email === 'boolean') next.email = parsed.email;

        return next;
      });
    } catch {
      // ignore
    }
  };

  // Hydrate persisted UI preferences on first load (optional)
  useEffect(() => {
    loadPersistedSettings();
  }, []);

  const update = <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const reset = () => {
    setSettings(defaultSettings);
    try {
      window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  const save = () => {
    try {
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // ignore
    }
  };

  // Theme is fixed: Dark UI + Purple brand color.
  const saveButtonTheme = { bg: 'bg-purple-600', hoverBg: 'hover:bg-purple-700' };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Settings</h1>
        <p className="text-muted-foreground mt-1">Configure camera, detection, alerts, and system behavior</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera Settings */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold text-foreground">Camera Settings</h2>
          <div className="mt-4 space-y-4">
            <div>
              <Label className="text-muted-foreground">Camera Source</Label>
              <Select value={settings.cameraSource} onValueChange={(v) => update('cameraSource', v as CameraSource)}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="webcam">Webcam</SelectItem>
                  <SelectItem value="video_file">Video File</SelectItem>
                  <SelectItem value="rtsp">RTSP</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-muted-foreground">Resolution</Label>
              <Select value={settings.resolution} onValueChange={(v) => update('resolution', v as Resolution)}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select resolution" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="640x480">640x480</SelectItem>
                  <SelectItem value="1280x720">1280x720</SelectItem>
                  <SelectItem value="1920x1080">1920x1080</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-muted-foreground">FPS</Label>
              <Input
                type="number"
                inputMode="numeric"
                value={Number.isFinite(settings.fps) ? settings.fps : ''}
                onChange={(e) => update('fps', Number(e.target.value))}
                className="mt-1"
              />
            </div>

            <div>
              <Label className="text-muted-foreground">RTSP URL</Label>
              <Input
                value={settings.rtspUrl}
                onChange={(e) => update('rtspUrl', e.target.value)}
                className="mt-1"
              />
            </div>
          </div>
        </Card>

        {/* Detection Settings */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold text-foreground">Detection Settings</h2>
          <div className="mt-4 space-y-6">
            <div>
              <Label className="text-muted-foreground">Confidence</Label>
              <div className="mt-2">
                <Slider
                  value={[settings.confidence]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={(v) => update('confidence', v[0] ?? 0)}
                />
              </div>
            </div>

            <div>
              <Label className="text-muted-foreground">Loitering threshold</Label>
              <Input
                type="number"
                inputMode="numeric"
                value={Number.isFinite(settings.loiteringThreshold) ? settings.loiteringThreshold : ''}
                onChange={(e) => update('loiteringThreshold', Number(e.target.value))}
                className="mt-1"
              />
            </div>

            <div>
              <Label className="text-muted-foreground">Bag threshold</Label>
              <Input
                type="number"
                inputMode="numeric"
                value={Number.isFinite(settings.bagThreshold) ? settings.bagThreshold : ''}
                onChange={(e) => update('bagThreshold', Number(e.target.value))}
                className="mt-1"
              />
            </div>

            <div>
              <Label className="text-muted-foreground">Detection toggles</Label>
              <div className="mt-2 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-foreground">People</span>
                  <Switch checked={settings.detectPeople} onCheckedChange={(v) => update('detectPeople', Boolean(v))} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-foreground">Loitering</span>
                  <Switch checked={settings.detectLoitering} onCheckedChange={(v) => update('detectLoitering', Boolean(v))} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-foreground">Running</span>
                  <Switch checked={settings.detectRunning} onCheckedChange={(v) => update('detectRunning', Boolean(v))} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-foreground">Unattended Bag</span>
                  <Switch
                    checked={settings.detectUnattendedBag}
                    onCheckedChange={(v) => update('detectUnattendedBag', Boolean(v))}
                  />
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Alert Settings */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold text-foreground">Alert Settings</h2>
          <div className="mt-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Sound toggle</span>
              <Switch checked={settings.sound} onCheckedChange={(v) => update('sound', Boolean(v))} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Alert logging toggle</span>
              <Switch checked={settings.alertLogging} onCheckedChange={(v) => update('alertLogging', Boolean(v))} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Email toggle</span>
              <Switch checked={settings.email} onCheckedChange={(v) => update('email', Boolean(v))} />
            </div>
          </div>
        </Card>

      </div>

      <div className="flex justify-end gap-3">
        <Button onClick={reset} variant="destructive" className="bg-red-600 hover:bg-red-700">
          Reset
        </Button>
        <Button onClick={save} className={`${saveButtonTheme.bg} ${saveButtonTheme.hoverBg} text-white`}>
          Save
        </Button>
      </div>
    </div>
  );
}
